// hipBLASLt solutions of one GEMM problem: list them, time them, validate them.
//
//   hipblaslt_sweep probe
//   hipblaslt_sweep list     m n k tA tB [--batch B] [--ws BYTES] [--heuristic N]
//   (every command also takes --types A,B,C and --compute TYPE)
//   hipblaslt_sweep time     m n k tA tB [--batch B] [--ws BYTES] [--rounds R] [--target_ms T]
//                                        [--warmup_ms W] [--mode rot|hot] INDEX...
//   hipblaslt_sweep validate m n k tA tB [--batch B] [--ws BYTES] INDEX...
//   hipblaslt_sweep screen   m n k tA tB [--batch B] [--ws BYTES] [--target_ms T] [--warmup_ms W]
//                                        [--mode rot|hot]
//
// Problems are column major, as hipBLASLt sees them: D[m, n] = op(A) op(B) + beta C
// with --types A,B,C (bf16_r, f16_r or f32_r; D has C's type, default bf16_r for
// all three), --compute f32_r (or xf32_r), f32 scale, alpha 1, --beta (default
// 0; XLA passes 1 when it adds a residual into the GEMM, with C and D one
// buffer, as here), and leading dimensions equal to the row counts. INDEX is a solution index, or -1 for the
// heuristic's first choice (what XLA runs at autotune level 0). A solution is
// supported if hipBLASLt accepts it for the problem within --ws bytes of
// workspace, the workspace XLA gives the call.
//
// Every result is one JSON object per line on stdout.
//   list:     {"kind": "heuristic", rank, index, workspace} for the first N
//             heuristic results, then {"kind": "solution", index, supported,
//             workspace, heuristic_rank, kernel, solution} for every solution.
//   time:     {"kind": "time", requested, index, round, calls, ms, tflops}.
//             Each round visits every index in a rotated order; each visit
//             times back-to-back calls for about --target_ms, after warm-up
//             calls, between two events. "rot" rotates over enough copies of
//             A, B and D to exceed 512 MiB, so caches don't hold them; "hot"
//             reuses one copy. --warmup_ms of the first index first brings the
//             GPU to its power-limited clock.
//   validate: {"kind": "validate", index, rel_l2, max_abs, default_rel_l2, ok}
//             against the same GEMM with an f32 D, from the heuristic.
//   screen:   {"kind": "screen", index, heuristic_rank, ms, tflops} for every
//             supported solution, each timed once for about --target_ms after
//             the warm-up, then {"kind": "screen_done", count, best, best_ms}.
//             A quick pass to pick finalists for `time`.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#define CHECK_HIP(x)                                                                     \
  do {                                                                                   \
    hipError_t e_ = (x);                                                                 \
    if (e_ != hipSuccess) {                                                              \
      fprintf(stderr, "HIP error %s at line %d\n", hipGetErrorString(e_), __LINE__);     \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)
#define CHECK_BLAS(x)                                                                    \
  do {                                                                                   \
    hipblasStatus_t e_ = (x);                                                            \
    if (e_ != HIPBLAS_STATUS_SUCCESS) {                                                  \
      fprintf(stderr, "hipBLASLt error %d at line %d\n", static_cast<int>(e_), __LINE__); \
      exit(1);                                                                           \
    }                                                                                    \
  } while (0)

namespace {

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

uint16_t to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

float from_bf16(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

struct Options {
  std::string command;
  int64_t m = 0, n = 0, k = 0, batch = 1;
  char ta = 'N', tb = 'N';
  size_t ws = 32ull << 20;
  int heuristic = 128, rounds = 5;
  double target_ms = 100, warmup_ms = 1000;
  std::string mode = "rot";
  std::string types = "bf16_r,bf16_r,bf16_r", compute = "f32_r";
  float beta = 0.f;
  std::vector<int> indices;
};

hipDataType data_type(const std::string& name) {
  if (name == "bf16_r") return HIP_R_16BF;
  if (name == "f16_r") return HIP_R_16F;
  if (name == "f32_r") return HIP_R_32F;
  fprintf(stderr, "unsupported type %s (bf16_r, f16_r, f32_r)\n", name.c_str());
  exit(2);
}

size_t type_bytes(hipDataType t) { return t == HIP_R_32F ? 4 : 2; }

hipblasComputeType_t compute_type(const std::string& name) {
  if (name == "f32_r") return HIPBLAS_COMPUTE_32F;
  if (name == "xf32_r") return HIPBLAS_COMPUTE_32F_FAST_TF32;
  fprintf(stderr, "unsupported compute type %s (f32_r, xf32_r)\n", name.c_str());
  exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  if (argc < 2) {
    fprintf(stderr, "usage: hipblaslt_sweep probe|list|screen|time|validate m n k tA tB [options] [INDEX...]\n");
    exit(2);
  }
  o.command = argv[1];
  if (o.command == "probe") return o;
  if (argc < 7) {
    fprintf(stderr, "%s needs m n k tA tB\n", o.command.c_str());
    exit(2);
  }
  o.m = atoll(argv[2]);
  o.n = atoll(argv[3]);
  o.k = atoll(argv[4]);
  o.ta = argv[5][0];
  o.tb = argv[6][0];
  for (int i = 7; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s needs a value\n", name);
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--batch") o.batch = atoll(next("--batch"));
    else if (a == "--ws") o.ws = strtoull(next("--ws"), nullptr, 10);
    else if (a == "--heuristic") o.heuristic = atoi(next("--heuristic"));
    else if (a == "--rounds") o.rounds = atoi(next("--rounds"));
    else if (a == "--target_ms") o.target_ms = atof(next("--target_ms"));
    else if (a == "--warmup_ms") o.warmup_ms = atof(next("--warmup_ms"));
    else if (a == "--mode") o.mode = next("--mode");
    else if (a == "--types") o.types = next("--types");
    else if (a == "--compute") o.compute = next("--compute");
    else if (a == "--beta") o.beta = static_cast<float>(atof(next("--beta")));
    else o.indices.push_back(atoi(a.c_str()));
  }
  return o;
}

// One problem's descriptors, operands and workspace.
struct Problem {
  Options o;
  hipblasLtHandle_t handle{};
  hipblasLtMatmulDesc_t desc{};
  hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
  hipblasLtMatmulPreference_t pref{};
  hipblasOperation_t op_a{}, op_b{};
  hipDataType type_a = HIP_R_16BF, type_b = HIP_R_16BF, type_c = HIP_R_16BF;
  hipblasComputeType_t compute = HIPBLAS_COMPUTE_32F;
  int64_t rows_a = 0, cols_a = 0, rows_b = 0, cols_b = 0;
  float alpha = 1.f, beta = 0.f;
  void* workspace = nullptr;
  hipStream_t stream{};

  explicit Problem(const Options& opts) : o(opts), beta(opts.beta) {
    CHECK_BLAS(hipblasLtCreate(&handle));
    {
      std::vector<std::string> t;
      size_t start = 0, comma;
      while ((comma = o.types.find(',', start)) != std::string::npos) {
        t.push_back(o.types.substr(start, comma - start));
        start = comma + 1;
      }
      t.push_back(o.types.substr(start));
      if (t.size() != 3) {
        fprintf(stderr, "--types needs A,B,C\n");
        exit(2);
      }
      type_a = data_type(t[0]);
      type_b = data_type(t[1]);
      type_c = data_type(t[2]);
      compute = compute_type(o.compute);
    }
    op_a = o.ta == 'T' ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    op_b = o.tb == 'T' ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    rows_a = o.ta == 'N' ? o.m : o.k;
    cols_a = o.ta == 'N' ? o.k : o.m;
    rows_b = o.tb == 'N' ? o.k : o.n;
    cols_b = o.tb == 'N' ? o.n : o.k;
    CHECK_BLAS(hipblasLtMatmulDescCreate(&desc, compute, HIP_R_32F));
    CHECK_BLAS(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    CHECK_BLAS(hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));
    la = layout(type_a, rows_a, cols_a);
    lb = layout(type_b, rows_b, cols_b);
    lc = layout(type_c, o.m, o.n);
    ld = layout(type_c, o.m, o.n);
    CHECK_BLAS(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t ws = o.ws;
    CHECK_BLAS(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws,
                                                     sizeof(ws)));
    CHECK_HIP(hipMalloc(&workspace, std::max<size_t>(o.ws, 256)));
    CHECK_HIP(hipStreamCreate(&stream));
  }

  hipblasLtMatrixLayout_t layout(hipDataType type, int64_t rows, int64_t cols) const {
    hipblasLtMatrixLayout_t l;
    CHECK_BLAS(hipblasLtMatrixLayoutCreate(&l, type, rows, cols, rows));
    if (o.batch > 1) {
      int32_t count = static_cast<int32_t>(o.batch);
      int64_t stride = rows * cols;
      CHECK_BLAS(hipblasLtMatrixLayoutSetAttribute(l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)));
      CHECK_BLAS(hipblasLtMatrixLayoutSetAttribute(l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride,
                                                   sizeof(stride)));
    }
    return l;
  }

  size_t elements_a() const { return static_cast<size_t>(rows_a * cols_a * o.batch); }
  size_t elements_b() const { return static_cast<size_t>(rows_b * cols_b * o.batch); }
  size_t elements_d() const { return static_cast<size_t>(o.m * o.n * o.batch); }
  double flops() const { return 2.0 * o.batch * o.m * o.n * o.k; }
  size_t bytes_a() const { return elements_a() * type_bytes(type_a); }
  size_t bytes_b() const { return elements_b() * type_bytes(type_b); }
  size_t bytes_d() const { return elements_d() * type_bytes(type_c); }

  void all_algos(std::vector<hipblasLtMatmulHeuristicResult_t>& all) const {
    CHECK_BLAS(hipblaslt_ext::getAllAlgos(handle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, op_a, op_b, type_a,
                                          type_b, type_c, type_c, compute, all));
  }

  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic(int count) const {
    std::vector<hipblasLtMatmulHeuristicResult_t> out(std::max(count, 1));
    int returned = 0;
    hipblasStatus_t s =
        hipblasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, ld, pref, count, out.data(), &returned);
    out.resize(s == HIPBLAS_STATUS_SUCCESS ? returned : 0);
    return out;
  }

  // The algorithm for `index` (-1: heuristic #0), and its workspace; false if unsupported.
  bool resolve(int index, hipblasLtMatmulAlgo_t* algo, int* resolved, size_t* ws_needed) {
    std::vector<hipblasLtMatmulHeuristicResult_t> found;
    if (index < 0) {
      found = heuristic(1);
    } else {
      std::vector<int> ids{index};
      if (hipblaslt_ext::getAlgosFromIndex(handle, ids, found) != HIPBLAS_STATUS_SUCCESS) found.clear();
    }
    if (found.empty()) return false;
    *algo = found[0].algo;
    *resolved = hipblaslt_ext::getIndexFromAlgo(*algo);
    size_t ws = 0;
    bool ok = hipblaslt_ext::matmulIsAlgoSupported(handle, desc, &alpha, la, lb, &beta, lc, ld, *algo, ws) ==
                  HIPBLAS_STATUS_SUCCESS &&
              ws <= o.ws;
    *ws_needed = ws;
    return ok;
  }

  void run(const hipblasLtMatmulAlgo_t& algo, const void* a, const void* b, void* d) {
    CHECK_BLAS(hipblasLtMatmul(handle, desc, &alpha, a, la, b, lb, &beta, d, lc, d, ld, &algo, workspace, o.ws,
                               stream));
  }
};

// Device copies of bf16 operands filled from one random pattern, at different offsets.
struct Operands {
  std::vector<void*> a, b, d;
  Operands(const Problem& p, int sets) {
    std::mt19937 gen(1);
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> pattern((1 << 22) + 7);
    for (auto& v : pattern) v = normal(gen);
    auto fill = [&](size_t count, size_t offset, hipDataType type) {
      size_t bytes = type_bytes(type);
      void* ptr;
      CHECK_HIP(hipMalloc(&ptr, count * bytes));
      std::vector<uint8_t> host(std::min<size_t>(count, 1 << 24) * bytes);
      size_t chunk_elems = host.size() / bytes;
      for (size_t done = 0; done < count; done += chunk_elems) {
        size_t chunk = std::min(chunk_elems, count - done);
        for (size_t i = 0; i < chunk; i++) {
          float f = pattern[(done + i + offset) % pattern.size()];
          if (type == HIP_R_32F) {
            memcpy(&host[i * 4], &f, 4);
          } else {
            uint16_t h = type == HIP_R_16BF ? to_bf16(f) : static_cast<uint16_t>(__half_as_ushort(__float2half(f)));
            memcpy(&host[i * 2], &h, 2);
          }
        }
        CHECK_HIP(hipMemcpy(static_cast<uint8_t*>(ptr) + done * bytes, host.data(), chunk * bytes, hipMemcpyHostToDevice));
      }
      return ptr;
    };
    for (int s = 0; s < sets; s++) {
      a.push_back(fill(p.elements_a(), 1000003ull * s, p.type_a));
      b.push_back(fill(p.elements_b(), 7777777ull + 1000003ull * s, p.type_b));
      void* ptr;
      CHECK_HIP(hipMalloc(&ptr, p.bytes_d()));
      CHECK_HIP(hipMemset(ptr, 0, p.bytes_d()));
      d.push_back(ptr);
    }
  }
  ~Operands() {
    for (auto* v : {&a, &b, &d})
      for (void* ptr : *v) (void)hipFree(ptr);
  }
};

int probe_command() {
  hipblasLtHandle_t handle;
  CHECK_BLAS(hipblasLtCreate(&handle));
  char revision[256] = {0};
  int version = 0;
  CHECK_BLAS(hipblasLtGetGitRevision(handle, revision));
  CHECK_BLAS(hipblasLtGetVersion(handle, &version));
  int device = 0;
  CHECK_HIP(hipGetDevice(&device));
  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, device));
  printf("{\"kind\": \"probe\", \"version\": %d, \"git_revision\": %s, \"gpu\": %s, \"gcn_arch\": %s, "
         "\"cus\": %d, \"clock_mhz\": %d, \"memory_gb\": %.1f}\n",
         version, json_string(revision).c_str(), json_string(prop.name).c_str(),
         json_string(prop.gcnArchName).c_str(), prop.multiProcessorCount, prop.clockRate / 1000,
         prop.totalGlobalMem / 1e9);
  hipblasLtDestroy(handle);
  return 0;
}

int list_command(Problem& p) {
  std::map<int, int> rank;
  auto heur = p.heuristic(p.o.heuristic);
  for (size_t i = 0; i < heur.size(); i++) {
    int index = hipblaslt_ext::getIndexFromAlgo(heur[i].algo);
    rank.emplace(index, static_cast<int>(i));
    printf("{\"kind\": \"heuristic\", \"rank\": %zu, \"index\": %d, \"workspace\": %zu}\n", i, index,
           heur[i].workspaceSize);
  }
  std::vector<hipblasLtMatmulHeuristicResult_t> all;
  p.all_algos(all);
  for (auto& r : all) {
    int index = hipblaslt_ext::getIndexFromAlgo(r.algo);
    size_t ws = 0;
    bool ok = hipblaslt_ext::matmulIsAlgoSupported(p.handle, p.desc, &p.alpha, p.la, p.lb, &p.beta, p.lc, p.ld,
                                                   r.algo, ws) == HIPBLAS_STATUS_SUCCESS;
    auto it = rank.find(index);
    printf("{\"kind\": \"solution\", \"index\": %d, \"supported\": %d, \"workspace\": %zu, "
           "\"heuristic_rank\": %d, \"kernel\": %s, \"solution\": %s}\n",
           index, ok && ws <= p.o.ws ? 1 : 0, ws, it == rank.end() ? -1 : it->second,
           json_string(hipblaslt_ext::getKernelNameFromAlgo(p.handle, r.algo)).c_str(),
           json_string(hipblaslt_ext::getSolutionNameFromAlgo(p.handle, r.algo)).c_str());
  }
  return 0;
}

struct Candidate {
  int requested = 0, index = -1;
  hipblasLtMatmulAlgo_t algo{};
  int calls = 1;
};

int time_command(Problem& p) {
  size_t set_bytes = p.bytes_a() + p.bytes_b() + p.bytes_d();
  int sets = p.o.mode == "hot" ? 1 : std::max(2, static_cast<int>(std::ceil(512.0 * (1 << 20) / set_bytes)));
  Operands ops(p, sets);
  std::vector<Candidate> cands;
  for (int requested : p.o.indices) {
    Candidate c;
    c.requested = requested;
    size_t ws = 0;
    if (!p.resolve(requested, &c.algo, &c.index, &ws)) {
      printf("{\"kind\": \"unsupported\", \"requested\": %d, \"index\": %d, \"workspace\": %zu}\n", requested,
             c.index, ws);
      continue;
    }
    cands.push_back(c);
  }
  if (cands.empty()) return 1;
  hipEvent_t start, stop;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  int cursor = 0;
  auto launch = [&](const Candidate& c, int calls) {
    for (int i = 0; i < calls; i++, cursor = (cursor + 1) % sets) p.run(c.algo, ops.a[cursor], ops.b[cursor], ops.d[cursor]);
  };
  auto timed = [&](const Candidate& c, int calls) {
    launch(c, 2);
    CHECK_HIP(hipEventRecord(start, p.stream));
    launch(c, calls);
    CHECK_HIP(hipEventRecord(stop, p.stream));
    CHECK_HIP(hipEventSynchronize(stop));
    float ms = 0;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms) / calls;
  };
  // Pilot each candidate to size its loop to about target_ms.
  for (auto& c : cands) {
    double per_call = timed(c, 5);
    c.calls = std::max(1, std::min(200000, static_cast<int>(p.o.target_ms / std::max(per_call, 1e-4))));
  }
  if (p.o.warmup_ms > 0) {
    double per_call = timed(cands[0], 3);
    launch(cands[0], std::max(1, static_cast<int>(p.o.warmup_ms / std::max(per_call, 1e-4))));
    CHECK_HIP(hipStreamSynchronize(p.stream));
  }
  for (int round = 0; round < p.o.rounds; round++) {
    for (size_t j = 0; j < cands.size(); j++) {
      const Candidate& c = cands[(j + round) % cands.size()];
      double ms = timed(c, c.calls);
      printf("{\"kind\": \"time\", \"requested\": %d, \"index\": %d, \"round\": %d, \"calls\": %d, \"ms\": %.6f, "
             "\"tflops\": %.2f}\n",
             c.requested, c.index, round, c.calls, ms, p.flops() / (ms * 1e-3) / 1e12);
      fflush(stdout);
    }
  }
  return 0;
}

int screen_command(Problem& p) {
  size_t set_bytes = p.bytes_a() + p.bytes_b() + p.bytes_d();
  int sets = p.o.mode == "hot" ? 1 : std::max(2, static_cast<int>(std::ceil(512.0 * (1 << 20) / set_bytes)));
  Operands ops(p, sets);
  std::map<int, int> rank;
  auto heur = p.heuristic(p.o.heuristic);
  for (size_t i = 0; i < heur.size(); i++) rank.emplace(hipblaslt_ext::getIndexFromAlgo(heur[i].algo), static_cast<int>(i));
  std::vector<hipblasLtMatmulHeuristicResult_t> all;
  p.all_algos(all);
  hipEvent_t start, stop;
  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  int cursor = 0;
  // Launches `calls` calls; false if hipBLASLt rejects the solution at run time.
  auto launch = [&](const hipblasLtMatmulAlgo_t& algo, int calls) {
    for (int i = 0; i < calls; i++, cursor = (cursor + 1) % sets) {
      if (hipblasLtMatmul(p.handle, p.desc, &p.alpha, ops.a[cursor], p.la, ops.b[cursor], p.lb, &p.beta,
                          ops.d[cursor], p.lc, ops.d[cursor], p.ld, &algo, p.workspace, p.o.ws,
                          p.stream) != HIPBLAS_STATUS_SUCCESS)
        return false;
    }
    return true;
  };
  auto timed = [&](const hipblasLtMatmulAlgo_t& algo, int calls, double* ms) {
    CHECK_HIP(hipEventRecord(start, p.stream));
    bool ok = launch(algo, calls);
    CHECK_HIP(hipEventRecord(stop, p.stream));
    CHECK_HIP(hipEventSynchronize(stop));
    float elapsed = 0;
    CHECK_HIP(hipEventElapsedTime(&elapsed, start, stop));
    *ms = static_cast<double>(elapsed) / calls;
    return ok;
  };
  if (!heur.empty() && p.o.warmup_ms > 0) {
    double per_call = 0;
    timed(heur[0].algo, 3, &per_call);
    launch(heur[0].algo, std::max(1, static_cast<int>(p.o.warmup_ms / std::max(per_call, 1e-4))));
    CHECK_HIP(hipStreamSynchronize(p.stream));
  }
  int count = 0, best = -1;
  double best_ms = 1e30;
  for (auto& r : all) {
    size_t ws = 0;
    if (hipblaslt_ext::matmulIsAlgoSupported(p.handle, p.desc, &p.alpha, p.la, p.lb, &p.beta, p.lc, p.ld, r.algo,
                                             ws) != HIPBLAS_STATUS_SUCCESS ||
        ws > p.o.ws)
      continue;
    int index = hipblaslt_ext::getIndexFromAlgo(r.algo);
    double per_call = 0, ms = 0;
    if (!timed(r.algo, 2, &per_call)) {
      printf("{\"kind\": \"screen_error\", \"index\": %d}\n", index);
      continue;
    }
    int calls = std::max(3, std::min(100000, static_cast<int>(p.o.target_ms / std::max(per_call, 1e-4))));
    timed(r.algo, calls, &ms);
    auto it = rank.find(index);
    printf("{\"kind\": \"screen\", \"index\": %d, \"heuristic_rank\": %d, \"ms\": %.6f, \"tflops\": %.2f}\n", index,
           it == rank.end() ? -1 : it->second, ms, p.flops() / (ms * 1e-3) / 1e12);
    count++;
    if (ms < best_ms) best_ms = ms, best = index;
  }
  printf("{\"kind\": \"screen_done\", \"count\": %d, \"best\": %d, \"best_ms\": %.6f}\n", count, best, best_ms);
  return 0;
}

std::vector<float> download(const void* ptr, size_t count, hipDataType type) {
  std::vector<float> out(count);
  if (type == HIP_R_32F) {
    CHECK_HIP(hipMemcpy(out.data(), ptr, count * 4, hipMemcpyDeviceToHost));
    return out;
  }
  std::vector<uint16_t> host(count);
  CHECK_HIP(hipMemcpy(host.data(), ptr, count * 2, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; i++)
    out[i] = type == HIP_R_16BF ? from_bf16(host[i]) : __half2float(__ushort_as_half(host[i]));
  return out;
}

void errors(const std::vector<float>& got, const std::vector<float>& ref, double* rel_l2, double* max_abs) {
  double num = 0, den = 0, worst = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    double d = static_cast<double>(got[i]) - ref[i];
    num += d * d;
    den += static_cast<double>(ref[i]) * ref[i];
    worst = std::max(worst, std::fabs(d));
  }
  *rel_l2 = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
  *max_abs = worst;
}

int validate_command(Problem& p) {
  Operands ops(p, 1);
  size_t count = p.elements_d();
  // Reference: the same GEMM with an f32 C and D.
  std::vector<float> ref;
  {
    hipblasLtMatrixLayout_t lc32 = p.layout(HIP_R_32F, p.o.m, p.o.n), ld32 = p.layout(HIP_R_32F, p.o.m, p.o.n);
    std::vector<hipblasLtMatmulHeuristicResult_t> h(1);
    int returned = 0;
    hipblasLtMatmulAlgoGetHeuristic(p.handle, p.desc, p.la, p.lb, lc32, ld32, p.pref, 1, h.data(), &returned);
    if (returned > 0) {
      void* d32;
      CHECK_HIP(hipMalloc(&d32, count * 4));
      CHECK_HIP(hipMemset(d32, 0, count * 4));
      CHECK_BLAS(hipblasLtMatmul(p.handle, p.desc, &p.alpha, ops.a[0], p.la, ops.b[0], p.lb, &p.beta, d32, lc32, d32,
                                 ld32, &h[0].algo, p.workspace, p.o.ws, p.stream));
      CHECK_HIP(hipStreamSynchronize(p.stream));
      ref.resize(count);
      CHECK_HIP(hipMemcpy(ref.data(), d32, count * 4, hipMemcpyDeviceToHost));
      (void)hipFree(d32);
    }
    hipblasLtMatrixLayoutDestroy(lc32);
    hipblasLtMatrixLayoutDestroy(ld32);
  }
  hipblasLtMatmulAlgo_t algo;
  int index = -1;
  size_t ws = 0;
  if (!p.resolve(-1, &algo, &index, &ws)) return 1;
  // C is D: zero it, so beta C adds nothing and every result is A B.
  CHECK_HIP(hipMemset(ops.d[0], 0, p.bytes_d()));
  p.run(algo, ops.a[0], ops.b[0], ops.d[0]);
  CHECK_HIP(hipStreamSynchronize(p.stream));
  std::vector<float> base = download(ops.d[0], count, p.type_c);
  bool against_default = ref.empty();
  if (against_default) ref = base;
  double base_rel = 0, base_abs = 0;
  errors(base, ref, &base_rel, &base_abs);
  for (int requested : p.o.indices) {
    int resolved = -1;
    if (!p.resolve(requested, &algo, &resolved, &ws)) {
      printf("{\"kind\": \"validate\", \"index\": %d, \"ok\": false, \"reason\": \"unsupported\"}\n", requested);
      continue;
    }
    CHECK_HIP(hipMemset(ops.d[0], 0, p.bytes_d()));
    p.run(algo, ops.a[0], ops.b[0], ops.d[0]);
    CHECK_HIP(hipStreamSynchronize(p.stream));
    std::vector<float> got = download(ops.d[0], count, p.type_c);
    double rel = 0, worst = 0;
    errors(got, ref, &rel, &worst);
    // bf16 output rounding alone gives a relative error of about 2^-9.
    bool ok = std::isfinite(rel) && rel <= std::max(1.5 * base_rel + 1e-4, against_default ? 4e-3 : 8e-3);
    printf("{\"kind\": \"validate\", \"index\": %d, \"rel_l2\": %.3e, \"max_abs\": %.3e, \"default_index\": %d, "
           "\"default_rel_l2\": %.3e, \"reference\": %s, \"ok\": %s}\n",
           resolved, rel, worst, index, base_rel, against_default ? "\"default\"" : "\"f32\"", ok ? "true" : "false");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options o = parse(argc, argv);
  if (o.command == "probe") return probe_command();
  Problem p(o);
  if (o.command == "list") return list_command(p);
  if (o.command == "time") return time_command(p);
  if (o.command == "validate") return validate_command(p);
  if (o.command == "screen") return screen_command(p);
  fprintf(stderr, "unknown command %s\n", o.command.c_str());
  return 2;
}

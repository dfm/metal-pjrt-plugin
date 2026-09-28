// The suffix of internal errors (a broken invariant of the plugin's own
// passes, not something the user's program can fix).
#ifndef METAL_PJRT_COMPILER_REPORT_BUG_H_
#define METAL_PJRT_COMPILER_REPORT_BUG_H_

namespace metal_pjrt {

inline constexpr char kReportBug[] =
    " (metal-pjrt-plugin bug; please report it with the HLO from "
    "XLA_FLAGS=--xla_dump_to=<dir>)";

}  // namespace metal_pjrt

#endif  // METAL_PJRT_COMPILER_REPORT_BUG_H_

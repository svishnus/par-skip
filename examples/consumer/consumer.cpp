// Smallest possible use of the installed package.
#include <cstdio>

#include "mskip/data.hpp"
#include "mskip/metric.hpp"
#include "mskip/metric_skip_list.hpp"

int main() {
  auto pts = mskip::data::uniform<2>(10000);
  mskip::MetricSkipList<mskip::L2<2>> S(pts, 4);
  S.build();
  const mskip::Neighbor nn = S.nearest_dist({0.5f, 0.5f});
  std::printf("nearest to (0.5, 0.5): point %u at distance %g\n", nn.index, double(nn.dist));
  return 0;
}

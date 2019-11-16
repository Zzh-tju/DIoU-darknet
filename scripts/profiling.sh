perf record --call-graph dwarf
git clone --depth 1 https://github.com/brendangregg/FlameGraph
cd FlameGraph
cp ../perf.data ./
perf script | ./stackcollapse-perf.pl |./flamegraph.pl > perf.svg

nvprof --analysis-metrics -o  nbody-analysis.nvprof ./nbody --benchmark -numdevices=2 -i=1

./nvprof --print-dependency-analysis-trace --dependency-analysis --cpu-thread-tracing on


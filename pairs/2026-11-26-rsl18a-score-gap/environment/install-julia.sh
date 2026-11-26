set -eu
cd "$HOME"
base=https://julialang-s3.julialang.org/bin
curl -fsSL --retry 2 "$base/linux/x64/1.12/julia-1.12.6-linux-x86_64.tar.gz" -o julia.tar.gz
curl -fsSL --retry 2 'https://julialang-s3.julialang.org/bin/checksums/julia-1.12.6.sha256' -o julia.sha256
want=$(awk '$2=="julia-1.12.6-linux-x86_64.tar.gz"{print $1}' julia.sha256)
test -n "$want"
echo "$want  julia.tar.gz" | sha256sum -c -
tar xzf julia.tar.gz
./julia-1.12.6/bin/julia --version

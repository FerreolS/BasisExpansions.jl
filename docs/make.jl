using Pkg

Pkg.develop(PackageSpec(path = dirname(@__DIR__)))

using Documenter
using NonLocalMeans

makedocs(
    modules = [BasisExpansions],
    sitename = "BasisExpansions.jl",
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/FerreolS/BasisExpansions.jl.git",
    devbranch = "master",
)

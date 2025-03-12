using TreeTCI
using Documenter

DocMeta.setdocmeta!(TreeTCI, :DocTestSetup, :(using TreeTCI); recursive = true)

makedocs(;
    modules = [TreeTCI],
    authors = "H. Shinaoka <h.shinaoka@gmail.com>",
    sitename = "TreeTCI.jl",
    format = Documenter.HTML(;
        canonical = "https://github.com/tensor4all/TreeTCI.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md"
    ],
)

deploydocs(; repo = "github.com/tensor4all/TreeTCI.jl.git", devbranch = "main")

using T4ATemplate
using Documenter

DocMeta.setdocmeta!(T4ATemplate, :DocTestSetup, :(using T4ATemplate); recursive=true)

makedocs(;
    modules=[T4ATemplate],
    authors="H. Shinaoka <h.shinaoka@gmail.com>",
    sitename="T4ATemplate.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/T4ATemplate.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo="github.com/tensor4all/T4ATemplate.jl.git",
    devbranch="main",
)

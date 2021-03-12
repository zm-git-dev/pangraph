module scRNA

include("io.jl")
using DataIO: read_mtx, read_barcodes, read_features

# ------------------------------------------------------------------------
# globals

BarcodeFile = "barcodes.tsv"
FeatureFile = "features.tsv"
CountMatrix = "matrix.mtx"

# ------------------------------------------------------------------------
# type w/ (de)serialization

struct Count{T} where T <: Real
    data    :: Array{T,2}
    gene    :: Array{String}
    barcode :: Array{String}
end

Count(data::Array{T,2}, gene, barcode) where T <: Real = Count{T}(data,gene,barcode)

# ------------------------------------------------------------------------
# operators

function scRNAload(dir::AbstractString)
    !isdir(dir) && error("directory '$dir' not found")

    files = listdir(dir)

    BarcodeFile ∉ files && error("'$BarcodeFile' not found in directory '$dir'")
    FeatureFile ∉ files && error("'$FeatureFile' not found in directory '$dir'")
    CountMatrix ∉ files && error("'$CountMatrix' not found in directory '$dir'")

    counts   = open(read_mtx,      "$dir/$CountMatrix")
    barcodes = open(read_barcodes, "$dir/$BarcodeFile")
    features = open(read_features, "$dir/$FeatureFile")

    length(barcodes) ≠ size(counts,2) && error("number of barcodes $(length(barcodes)) ≠ number of columns $(size(counts,2)). check data")
    length(features) ≠ size(counts,1) && error("number of features $(length(features)) ≠ number of rows $(size(counts,1)). check data")

    return Count(data,features,barcodes)
end

end

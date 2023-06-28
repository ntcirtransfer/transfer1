import os
import ir_datasets
from ir_datasets.formats import JsonlDocs, JsonlQueries, TrecQrels, TrecScoredDocs
from ir_datasets.datasets.base import Dataset, YamlDocumentation, FilteredQueries, Deprecated

# A unique identifier for this dataset. To avoid name conflicts, consider prefixing
# identifiers with a person/org, as done here:
NAME = 'ntcir-transfer'

# What to the relevance levels in qrels mean?
QREL_DEFS_TRAIN = {
    2: 'relevant',
    1: 'partially relevant',
	0: 'not relevant',
}

def _init():
    # where the content is cached
    base_path = ir_datasets.util.home_path() / NAME
    
    # Specify where to find the content. Here it's just from the repository, but it could be anywhere.
    TC_ROOT = os.path.join(os.path.dirname(os.path.abspath('__file__')), '..')
    DL_DOCS_TRAIN = ir_datasets.util.LocalDownload(TC_ROOT + '/testcollections/ntcir/NTCIR-1/mlir/ntc1-j1.utf8.jsonl')
    DL_QUERIES_TRAIN = ir_datasets.util.LocalDownload(TC_ROOT + '/testcollections/ntcir/NTCIR-1/topics/topic0001-0083.utf8.jsonl')
    DL_QRELS_TRAIN = ir_datasets.util.LocalDownload(TC_ROOT + '/testcollections/ntcir/NTCIR-1/mlir/rel2_ntc1-j1_0001-0083.utf8.tsv')
    DL_TOP1000_TRAIN = ir_datasets.util.LocalDownload(TC_ROOT + '/testcollections/ntcir/NTCIR-1/mlir/top1000.train.tsv')

    DL_DOCS_EVAL = ir_datasets.util.LocalDownload(TC_ROOT + '/testcollections/ntcir/NTCIR-2/j-docs/ntc12-j1gk.mod.jsonl')
    DL_QUERIES_EVAL = ir_datasets.util.LocalDownload(TC_ROOT + '/testcollections/ntcir/NTCIR-2/topics/topic-j0101-0149.utf8.jsonl')
    DL_TOP1000_EVAL = ir_datasets.util.LocalDownload(TC_ROOT + '/testcollections/ntcir/NTCIR-2/j-docs/top1000.eval.tsv')
    
    subsets = {}

    # Dataset definition: it provides docs, queries, and qrels
    subsets['1/train'] = ir_datasets.Dataset(
        JsonlDocs(ir_datasets.util.Cache(DL_DOCS_TRAIN, base_path/'ntc1-j1.utf8.jsonl')),
        JsonlQueries(ir_datasets.util.Cache(DL_QUERIES_TRAIN, base_path/'topic0001-0083.utf8.jsonl')),
        TrecQrels(ir_datasets.util.Cache(DL_QRELS_TRAIN, base_path/'rel2_ntc1-j1_0001-0083.utf8.tsv'), QREL_DEFS_TRAIN),
        TrecScoredDocs(ir_datasets.util.Cache(DL_TOP1000_TRAIN, base_path/'top1000.train.tsv')),
    )
    subsets['1/eval'] = ir_datasets.Dataset(
        JsonlDocs(ir_datasets.util.Cache(DL_DOCS_EVAL, base_path/'ntc12-j1gk.mod.jsonl')),
        JsonlQueries(ir_datasets.util.Cache(DL_QUERIES_EVAL, base_path/'topic-j0101-0149.utf8.jsonl')),
        TrecScoredDocs(ir_datasets.util.Cache(DL_TOP1000_EVAL, base_path/'top1000.eval.tsv')),
    )

    # Register the dataset with ir_datasets
    documentation = YamlDocumentation(f'{NAME}.yaml')
    base = Dataset(documentation('_'))
    ir_datasets.registry.register(f'{NAME}', base)
    for s in subsets:
        ir_datasets.registry.register(f'{NAME}/{s}', subsets[s])
    
    return base, subsets


base, subsets = _init()

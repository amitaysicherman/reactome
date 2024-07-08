from common.args_manager import get_args
from dataset.index_manger import get_from_args, NodesIndexManager
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# args = get_args()
index_manager: NodesIndexManager = NodesIndexManager(pretrained_method=2, fuse_name="KIBA_ProtBertT5-xl_roberta",
                                                     prot_emd_type="ProtBertT5-xl")
proteins = index_manager.protein_array
proteins_indexes = index_manager.protein_indexes
molecules = index_manager.molecule_array
molecules_indexes = index_manager.molecule_indexes
texts = index_manager.text_array
texts_indexes = index_manager.text_indexes

min_dim=min(proteins.shape[1], molecules.shape[1], texts.shape[1])
if proteins.shape[1] > min_dim:
   proteins=PCA(n_components=min_dim).fit_transform(proteins)
if molecules.shape[1] > min_dim:
    molecules=PCA(n_components=min_dim).fit_transform(molecules)
if texts.shape[1] > min_dim:
    texts=PCA(n_components=min_dim).fit_transform(texts)

concat_array=np.concatenate([proteins, molecules, texts])
#normalize all vectors to length 1
concat_array=concat_array/np.linalg.norm(concat_array, axis=1, keepdims=True)
array_2d=TSNE(n_components=2).fit_transform(concat_array)
proteins_2d=array_2d[:proteins.shape[0]]
molecules_2d=array_2d[proteins.shape[0]:proteins.shape[0]+molecules.shape[0]]
texts_2d=array_2d[proteins.shape[0]+molecules.shape[0]:]
plt.scatter(proteins_2d[:, 0], proteins_2d[:, 1], label="Proteins")
plt.scatter(molecules_2d[:, 0], molecules_2d[:, 1], label="Molecules")
plt.scatter(texts_2d[:, 0], texts_2d[:, 1], label="Texts")

plt.legend()
plt.show()

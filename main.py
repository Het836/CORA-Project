"""
We are going to create our API now. 
"""
from typing import List, Optional
from fastapi import FastAPI, HTTPException
import os
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
      

"""
The model can predict 7 possible topics for each paper.
Their is a dictionary that checks which label belongs to which topic.
"""
CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Networks",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}

"""
Figure out where this file lives, so we can find the model file
"""
BASE_DIR    = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE_DIR, 'simple_gcn_cora.onnx')
DATA_DIR    = os.path.join(BASE_DIR,"data", "Planetoid")
STATIC_DIR = os.path.join(BASE_DIR, 'static')

"""
Load the train model once, when the app states up
"""
model_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])


app = FastAPI()

"""
Request formats: these describes what JSON data user must send us
"""
class GraphPredictRequest(BaseModel):
    # A list of nodes, where each nodes is a list of 1433 numbers(its features)
    node_features : List[List[float]]

    # Optinal : Whicg nodes connect to which shape[2, num of edges]
    edge_indices : Optional[List[List[int]]] = None

class CoraNodeRequest(BaseModel):
    # Which node (by index number) from the real cora dataset to classify
    node_indices: List[int]


"""
Helper Function
"""
def softmax(scores: np.ndarray):
    # Turn raw model scores (logits) into probablites that comes between 0 to 1.
    # We will get the max probability
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

def run_model(node_features: np.ndarray, edge_index: np.ndarray, node_indices_to_return):
    """
    Run ONNX model only on the graph we are given and return its predictions.
    """
    output = model_session.run(
        ['logits'],
        {
            'node_features': node_features.astype(np.float32),
            'edge_indices': edge_index.astype(np.int64)
        },
    )
    logits = output[0]
    probabilities = softmax(logits)
    predicted_classes = logits.argmax(axis=-1)

    results = []
    for i in node_indices_to_return:
        results.append({
            'node_index'                : i,
            'predicted_classes_id'      : int(predicted_classes[i]),
            'predicted_classes_name'    : CORA_CLASSES[int(predicted_classes[i])],
            'probabilities'             : probabilities[i].tolist(),
            'logits'                    : logits[i].tolist(),
        })

    return {
        'num_nodes': node_features.shape[0],
        'num_edges': edge_index.shape[1],
        'predictions': results
    }


"""
Routes
"""

@app.get('/')
def home_page():
    """
    Our UI will be shown
    """
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"service": "Simple GCN Cora API", "status": "Running"}


@app.get('/health')
def health_check():
    return {'status': 'healthy', 'provider': model_session.get_providers()}

def model_info():
    """
    Shows basic details about the model inputs, outputs, classes
    """
    return {
        'Model Name': 'SimpleGCN',
        'feature_dimension': 1433,
        'num_classes': 7,
        'class_mapping': CORA_CLASSES,
        'inputs': [
            {'name': inp.name, 'shape': inp.shape, 'type':inp.type}
            for inp in model_session.get_inputs()
        ],
        'outputs': [
            {'name': out.name, 'shape': out.shape, 'type':out.type}
            for out in model_session.get_outputs()
        ],
    }


@app.post('/predict')
def predict_custom_graph(request: GraphPredictRequest):
    """
    Classify a graph you provide yourself and optionally your own edge/connections
    """
    if not request.node_features or request.node_features == 0:
        raise HTTPException(400, "Node Features cannot be empy")

    for feature_vector in request.node_features:
        if len(feature_vector) != 1433:
            raise HTTPException(422, "Each node's feature vector must exactly have 1433 features")

    node_features = np.arra(request.node_features, dtype=np.float32)
    num_nodes = len(request.node_features)

    if request.edge_indices:
        edge_index = np.array(request.edge_indices, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(422, "edge_indices must have shape [2, num_edges]")

    else:
        node_ids = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack([node_ids, node_ids])

    all_node_features = list(range(num_nodes))
    return run_model(node_features, edge_index, all_node_features)

@app.post('/predict/cora_node')
def predict_real_cora_nodes(request: CoraNodeRequest):
    """
    CLassify real papers from the actual Cora dataset
    """
    from torch_geometric.datasets import Planetoid
    try:
        cora_dataset = Planetoid(root=DATA_DIR, name="Cora")[0]
    except Exception as error:
        raise HTTPException(500, f"failed to load cora dataset: {error}")

    largest_valid_index = cora_dataset.num_nodes - 1
    invalid_index = [
        i for i in request.node_indices
        if i < 0 or i > largest_valid_index
    ]
    if invalid_index:
        raise HTTPException(
            400, f"node_index out of bound (must be 0 to {largest_valid_index})"
        )

    return run_model(
        cora_dataset.x.numpy(),
        cora_dataset.edge_index.numpy(),
        request.node_indices
    )


if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name='static')
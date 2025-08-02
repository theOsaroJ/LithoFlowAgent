from langchain.tools import BaseTool
from data_tools.load_and_map import load_and_map_all
from data_tools.preprocess import align_depths, flag_outliers
from model_tools.train_imputer import train_imputer
from model_tools.train_litho_classifier import train_litho_classifier
from model_tools.train_forecaster import train_forecaster
from retrieval.query_index import Retriever

class IngestLogsTool(BaseTool):
    name: str = "ingest_logs"
    description: str = "Load, normalize, align, and flag outliers from well-log files."
    def _run(self, directory: str):
        logs = load_and_map_all(directory)
        df = align_depths(list(logs.values()))
        return flag_outliers(df)

class ImputeLogsTool(BaseTool):
    name: str = "impute_logs"
    description: str = "Train XGBoost imputer for missing log curves."
    def _run(self, args: dict):
        train_imputer(args["df"], args["target_cols"], args["feature_cols"], args["out_path"])
        return f"Imputer saved to {args['out_path']}"

class LithoClassifierTool(BaseTool):
    name: str = "train_litho"
    description: str = "Train XGBoost classifier for lithofacies."
    def _run(self, args: dict):
        train_litho_classifier(args["df"], args["label_col"], args["feature_cols"], args["out_path"])
        return f"Lithology classifier saved to {args['out_path']}"

class ForecastTool(BaseTool):
    name: str = "train_forecaster"
    description: str = "Train LSTM forecaster on production data."
    def _run(self, args: dict):
        train_forecaster(
            args["data"], args.get("seq_len",12),
            args.get("lr",1e-3), args.get("epochs",20),
            args["save_path"]
        )
        return f"Forecaster saved to {args['save_path']}"

class RetrieveDocsTool(BaseTool):
    name: str = "retrieve_docs"
    description: str = "Retrieve relevant domain documents for a query."
    def _run(self, query: str):
        return "\n".join(f"{p} (score {s:.2f})" for p,s in Retriever("retrieval/faiss.index").retrieve(query))

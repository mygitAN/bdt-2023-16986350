"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    read_data,
    join_data,
    split_data,
    preprocess_data,
    train_model,
    predict,
    evaluate_model,
    feature_importance
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(           
                func=read_data,
                inputs=None,
                outputs=["df_internal_data", "df_external_data"],
                name="read_data",
            ),
            node(           
                func=join_data,
                inputs=["df_internal_data", "df_external_data"],
                outputs="df_select",
                name="join_data",
            ),
            node(           
                func=split_data,
                inputs=["df_select"],
                outputs=["train", "test"],
                name="split_data",
            ),
            node(           
                func=preprocess_data,
                inputs=["train", "test"],
                outputs=["train_data", "test_data", "assemblerInputs"],
                name="preprocess_data",
            ),
            node(           
                func=train_model,
                inputs="train_data",
                outputs="rfModel",
                name="train_model",
            ),
            node(           
                func=predict,
                inputs=["rfModel", "test_data"],
                outputs="predictions",
                name="predict",
            ),
            node(           
                func=evaluate_model,
                inputs="predictions",
                outputs=None,
                name="evaluate_model",
            ),
            node(           
                func=feature_importance,
                inputs=["rfModel", "assemblerInputs"],
                outputs=None,
                name="feature_importance"
            )
        ]
    )


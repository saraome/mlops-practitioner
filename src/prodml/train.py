import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from prodml.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from prodml.data import download_data, load_data, prepare_data
from prodml.features import add_features, prepare_feature_dicts


def main() -> None:
    # Make sure the dataset exists
    data_path = download_data()

    # Load and clean the data
    df = load_data(data_path)
    df = prepare_data(df)

    # Create PU_DO
    df = add_features(df)

    # Same train/validation split used in the notebook
    df_train, df_val = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Convert features to dictionaries
    train_dicts = prepare_feature_dicts(df_train)
    val_dicts = prepare_feature_dicts(df_val)

    # Fit DictVectorizer ONLY on training data
    dv = DictVectorizer()

    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    # Target
    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    # Same baseline model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validation predictions
    y_pred = model.predict(X_val)

    rmse = root_mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"Validation RMSE: {rmse:.3f}")
    print(f"Validation MAE: {mae:.3f}")
    print(f"R²: {r2:.3f}")

    # Save vectorizer and model together
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(
            {
                "vectorizer": dv,
                "model": model,
            },
            f,
        )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

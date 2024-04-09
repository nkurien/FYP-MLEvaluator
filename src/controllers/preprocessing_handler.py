from data_processing.preprocessing import CombinedPreprocessor, PreprocessingPipeline, SimpleImputer, OneHotEncoder, NumericConverter, MinMaxScaler, OrdinalEncoder

class PreprocessingHandler:
    def __init__(self):
        self.X = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.ordinal_columns = None
        self.combined_preprocessor = None

    def set_data(self, X):
        self.X = X

    def set_column_indices(self, categorical_columns, numerical_columns, ordinal_columns):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.ordinal_columns = ordinal_columns

    def create_preprocessor(self):
        try:
            # Create the preprocessor pipelines
            categorical_pipeline = PreprocessingPipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder())
            ])

            numerical_pipeline = PreprocessingPipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("converter", NumericConverter()),
                ("scaler", MinMaxScaler())
            ])

            ordinal_pipeline = PreprocessingPipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder())
            ])

            # Create the CombinedPreprocessor instance
            self.combined_preprocessor = CombinedPreprocessor(
                num=(numerical_pipeline, self.numerical_columns),
                cat=(categorical_pipeline, self.categorical_columns),
                ord=(ordinal_pipeline, self.ordinal_columns)
            )

            return self.combined_preprocessor

        except Exception as e:
            raise ValueError(str(e))
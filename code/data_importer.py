class CustomDataset:
    def __init__(
        self,
        name: str,
        train: List[pd.DataFrame],
        test: List[pd.DataFrame],
        label_list: List[str],
    ):
        """Class to hold and structure datasets.

        Args:

        name (str): holds the name of the dataset so we can select it later
        train (List[pd.DataFrame]): holds training pandas dataframe with 2 columns ["text","label"]
        test (List[pd.DataFrame]): holds testing pandas dataframe with 2 columns ["text","label"]
        label_list (List[str]): holds the list  of labels
        """
        self.name = name
        self.train = train
        self.test = test
        self.label_list = label_list
        
        
def read_HARD(file_path = "/content/balanced-reviews.txt"):
    
    df_HARD = pd.read_csv(file_path, sep="\t", header=0,encoding='utf-16')
    df_HARD = df_HARD[["review","rating"]]  # we are interested in rating and review only
    df_HARD.columns = [DATA_COLUMN, LABEL_COLUMN]
    print(df_HARD[LABEL_COLUMN].value_counts())
    # code rating as +ve if > 3, -ve if less, no 3s in dataset

    hard_map = {
        5: 'POS',
        4: 'POS',
        2: 'NEG',
        1: 'NEG'
    }

    df_HARD[LABEL_COLUMN] = df_HARD[LABEL_COLUMN].apply(lambda x: hard_map[x])
    train_HARD, test_HARD = train_test_split(df_HARD, test_size=0.2, random_state=42)
    label_list_HARD = ['NEG', 'POS']

    data_Hard = CustomDataset("HARD", train_HARD, test_HARD, label_list_HARD)
    return data_HARD


def read_ArSAS(file_path="/content/ArSAS..txt"):
    df_ArSAS = pd.read_csv(file_path, sep="\t",encoding='utf-8')
    df_ArSAS = df_ArSAS[["Tweet_text","Sentiment_label"]]  # we are interested in rating and review only
    df_ArSAS.columns = [DATA_COLUMN, LABEL_COLUMN]
    print("Total length: ", len(df_ArSAS))
    print(df_ArSAS[LABEL_COLUMN].value_counts())

    label_list_ArSAS = list(df_ArSAS[LABEL_COLUMN].unique())
    print(label_list_ArSAS)

    train_ArSAS, test_ArSAS = train_test_split(df_ArSAS, test_size=0.2, random_state=42)
    print("Training length: ", len(train_ArSAS))
    print("Testing length: ", len(test_ArSAS))
    data_ArSAS = CustomDataset("ArSAS", train_ArSAS, test_ArSAS, label_list_ArSAS)
    return data_ArSAS



def read_LABR():
    from labr import LABR
    labr_helper = LABR()

    (d_train, y_train, d_test, y_test) = labr_helper.get_train_test(
        klass="2", balanced="unbalanced"
    )

    train_LABR_B_U = pd.DataFrame({DATA_COLUMN: d_train, LABEL_COLUMN: y_train})
    test_LABR_B_U = pd.DataFrame({DATA_COLUMN: d_test, LABEL_COLUMN: y_test})

    train_LABR_B_U[LABEL_COLUMN] = train_LABR_B_U[LABEL_COLUMN].apply(lambda x: 'NEG' if (x == 0) else 'POS')
    test_LABR_B_U[LABEL_COLUMN] = test_LABR_B_U[LABEL_COLUMN].apply(lambda x: 'NEG' if (x == 0) else 'POS')

    print(train_LABR_B_U[LABEL_COLUMN].value_counts() + test_LABR_B_U[LABEL_COLUMN].value_counts())
    label_list_LABR_B_U = list(test_LABR_B_U[LABEL_COLUMN].unique())

    data_LABR_B_U = CustomDataset(
        "LABR-UN-Binary", train_LABR_B_U, test_LABR_B_U, label_list_LABR_B_U
    )
    all_datasets.append(data_LABR_B_U)
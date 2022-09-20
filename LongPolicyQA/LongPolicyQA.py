import datasets
import json


def make_dataset_use_entire_document_as_context(row):
    """
    Convert the paragraphs into documents by:
    1. Iteratre over each paragraph in the document (supplied row)
    2. Concatenate all of the paragraphs' contexts together to form a single document
    3. Update the 'context' field of each paragraph to be the document
    4. Update the start of each answer to be the start of the answer in the document
    :param row: a row from the dataset representing a single document
    :return: the updated dataset.
    """

    # join all of the contexts together to form a single document, separating each with a newline
    entire_document = "\n".join(
        [paragraph["context"] for paragraph in row["paragraphs"]]
    )

    # the offset relative to the start of the document
    context_offset = 0

    for paragraph in row["paragraphs"]:
        original_context = paragraph["context"]
        paragraph["context"] = entire_document
        for qa in paragraph["qas"]:
            for answer in qa["answers"]:
                answer["answer_start"] += context_offset

        # update the context offset for the next paragraph
        context_offset += len(original_context) + 1  # +1 for the newline character

    return row


def check_answer_offsets(row):
    """
    Check that the answer offsets are correct
    :param row: a row from the dataset representing a single document
    :return: Nothing
    """
    for paragraph in row["paragraphs"]:
        for qa in paragraph["qas"]:
            for answer in qa["answers"]:
                start = answer["answer_start"]
                selected_from_context = paragraph["context"][
                    start : start + len(answer["text"])
                ]
                actual_text = answer["text"]
                assert (
                    selected_from_context == actual_text
                ), f"Expected {actual_text} but got {selected_from_context}"


_URL = "https://raw.githubusercontent.com/wasiahmad/PolicyQA/main/data/"
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
    "test": _URL + "test.json",
}


class LongPolicyQAConfig(datasets.BuilderConfig):
    """BuilderConfig for LongPolicyQA."""

    def __init__(self, **kwargs):
        """BuilderConfig for LongPolicyQA..
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LongPolicyQAConfig, self).__init__(**kwargs)


class LongPolicyQA(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        LongPolicyQAConfig(
            name="long_policy_qa",
            version=datasets.Version("1.0.0"),
            description="PolicyQA dataset, with the entire document as context.",
        ),
    ]

    def _info(self):
        # TODO(squad_v2): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            task_templates=[
                datasets.tasks.QuestionAnsweringExtractive(
                    question_column="question",
                    context_column="context",
                    answers_column="answers",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(squad_v2): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        # update the downloaded files to have the entire context
        # for split, filepath in downloaded_files.items():
        #     print(f"Updating {filepath} ({split}) to have the entire context")
        #     with open(filepath, "r+") as f:
        #         dataset = json.load(f)
        #         print(f"Loaded {filepath} with {len(dataset['data'])} documents")
        #         for example in dataset["data"]:
        #             example_with_entire_context = (
        #                 make_dataset_use_entire_document_as_context(example)
        #             )
        #             check_answer_offsets(example_with_entire_context)
        #             example["paragraphs"] = example_with_entire_context["paragraphs"]
        #             print(
        #                 f"Updated {filepath} with {len(example['paragraphs'])} paragraphs"
        #             )

        #         print(f"writing to {filepath} ({split})")
        #         json.dump(dataset, f)
        #         print(f"wrote to {filepath} ({split})")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_files["test"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(squad_v2): Yields (key, example) tuples from the dataset
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for example in squad["data"]:
                title = example.get("title", "")
                example_with_entire_context = (
                    make_dataset_use_entire_document_as_context(example)
                )
                for paragraph in example_with_entire_context["paragraphs"]:
                    context = paragraph[
                        "context"
                    ]  # do not strip leading blank spaces GH-2585
                    for qa in paragraph["qas"]:
                        question = qa["question"]
                        id_ = qa["id"]

                        answer_starts = [
                            answer["answer_start"] for answer in qa["answers"]
                        ]
                        answers = [answer["text"] for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }

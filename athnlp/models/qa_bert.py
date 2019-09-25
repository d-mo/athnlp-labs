from typing import Dict, Optional

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import Average, BooleanAccuracy, CategoricalAccuracy, SquadEmAndF1
from allennlp.nn.util import masked_softmax, sequence_cross_entropy_with_logits
from allennlp.models.reading_comprehension.util import get_best_span

from overrides import overrides
from pytorch_transformers.modeling_bert import BertModel


@Model.register("qa_bert")
class BertQuestionAnswering(Model):
    """
    A QA model for SQuAD based on the AllenNLP Model ``BertForClassification`` that runs pretrained BERT,
    takes the pooled output, adds a Linear layer on top, and predicts two numbers: start and end span.

    Note that this is a somewhat non-AllenNLP-ish model architecture,
    in that it essentially requires you to use the "bert-pretrained"
    token indexer, rather than configuring whatever indexing scheme you like.
    See `allennlp/tests/fixtures/bert/bert_for_classification.jsonnet`
    for an example of what your config might look like.
    Parameters
    ----------
    vocab : ``Vocabulary``
    bert_model : ``Union[str, BertModel]``
        The BERT model to be wrapped. If a string is provided, we will call
        ``BertModel.from_pretrained(bert_model)`` and use the result.
    num_labels : ``int``, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the ``label_namespace``.
    index : ``str``, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : ``str``, optional (default : "labels")
        Used to determine the number of classes if ``num_labels`` is not supplied.
    trainable : ``bool``, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    initializer : ``InitializerApplicator``, optional
        If provided, will be used to initialize the final linear layer *only*.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 bert_model: BertModel,
                 dropout: float = 0.0,
                 index: str = "bert",
                 trainable: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None, ) -> None:
        super().__init__(vocab, regularizer)

        self._index = index
        self.bert_model = PretrainedBertModel.load(bert_model)
        hidden_size = self.bert_model.config.hidden_size

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        # 1. Instantiate any additional parts of your network
        self.drop = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(hidden_size, 2)
        
        # 2. DON'T FORGET TO INITIALIZE the additional parts of your network.
        initializer(self.linear)

        # 3. Instantiate your metrics
        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = SquadEmAndF1()

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self,  # type: ignore
                metadata: Dict,
                tokens: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField`` (that has a bert-pretrained token indexer)
        span_start : torch.IntTensor, optional (default = None)
            A tensor of shape (batch_size, 1) which contains the start_position of the answer
            in the passage, or 0 if impossible. This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : torch.IntTensor, optional (default = None)
            A tensor of shape (batch_size, 1) which contains the end_position of the answer
            in the passage, or 0 if impossible. This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        start_probs: torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        end_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        best_span:
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        input_ids = tokens[self._index]
        token_type_ids = tokens[f"{self._index}-type-ids"]
        input_mask = (input_ids != 0).long()

        # 1. Build model here
        bert_output, _ = self.bert_model(input_ids, token_type_ids, attention_mask=input_mask)
        linear_output = self.linear(bert_output)
        linear_dropped = self.drop(linear_output)
        start_logits, end_logits = linear_dropped.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        # 2. Compute start_position and end_position and then get the best span
        # using allennlp.models.reading_comprehension.util.get_best_span()
        masked_soft_start = masked_softmax(start_logits, mask=input_mask)
        masked_soft_end = masked_softmax(end_logits, mask=input_mask)
        best_span = get_best_span(masked_soft_start, masked_soft_end)
        output_dict = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_probs": masked_soft_start,
            "end_probs": masked_soft_end,
            "best_span": best_span
        }

        # 4. Compute loss and accuracies. You should compute at least:
        # span_start accuracy, span_end accuracy and full span accuracy.
        # import ipdb;ipdb.set_trace()
        self._span_start_accuracy(start_logits, span_start.squeeze())
        self._span_end_accuracy(end_logits, span_end.squeeze())
        self._span_accuracy(best_span,  torch.stack([span_start.squeeze(), span_end.squeeze()]))
        

        # UNCOMMENT THIS LINE
        # import ipdb;ipdb.set_trace()
        if span_start is not None:
            ignored_index = start_logits.size(1)
            span_start.clamp_(0, ignored_index)
            span_end.clamp_(0, ignored_index)
            start_loss = self.loss(start_logits, span_start.squeeze(-1))
            end_loss = self.loss(end_logits, span_end.squeeze(-1))
            combined_loss = (start_loss + end_loss) / 2
            output_dict["loss"] = combined_loss

        # 5. Optionally you can compute the official squad metrics (exact match, f1).
        # Instantiate the metric object in __init__ using allennlp.training.metrics.SquadEmAndF1()
        # When you call it, you need to give it the word tokens of the span (implement and call decode() below)
        # and the gold tokens found in metadata[i]['answer_texts']

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        pass

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # UNCOMMENT if you want to report official SQuAD metrics
        # exact_match, f1_score = self._squad_metrics.get_metric(reset)

        metrics = {
            'start_acc': self._span_start_accuracy.get_metric(reset),
            'end_acc': self._span_end_accuracy.get_metric(reset),
            'span_acc': self._span_accuracy.get_metric(reset),
            # 'em': exact_match,
            # 'f1': f1_score,
        }
        return metrics


class PretrainedBertModel:
    """
    In some instances you may want to load the same BERT model twice
    (e.g. to use as a token embedder and also as a pooling layer).
    This factory provides a cache so that you don't actually have to load the model twice.
    """
    _cache: Dict[str, BertModel] = {}

    @classmethod
    def load(cls, model_name: str, cache_model: bool = True) -> BertModel:
        if model_name in cls._cache:
            return PretrainedBertModel._cache[model_name]

        model = BertModel.from_pretrained(model_name)
        if cache_model:
            cls._cache[model_name] = model

        return model

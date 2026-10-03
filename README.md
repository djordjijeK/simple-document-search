# Simple Document Ranking - TF-IDF and BM25 on SciFact

This project builds a small search engine in Python to explore how **TF-IDF** and **BM25** rank documents. Given a query, both methods look for matching terms in a collection of documents and assign each document a score. The project uses the SciFact collection: scientific claims serve as queries, and the titles and abstracts of scientific papers form the searchable documents. The aim is to retrieve relevant papers; the search scores do not establish whether a claim is true.

The core tokenization, indexing, and ranking logic is implemented from scratch so that each step is easy to inspect. Both retrievers share the same documents, tokenizer, and inverted index, making it possible to compare their rankings. The implementation also includes evaluation using relevance judgments and Recall@k, MRR@k, and nDCG@k.

## How text becomes searchable

A **corpus** is the collection of documents we search. In this project, each document combines a paper's title and abstract. A **query** is the text we want to find relevant documents for.

Before scoring, we pass both document text and query text through the same **tokenizer**. It turns text into a sequence of tokens. A token is one occurrence in that sequence; a term is a distinct token value that we can count and match.

For example:

```text
Query:    "Gene therapy"
Tokens:   ["gene", "therapy"]

Document: "Gene therapy studies examine gene expression."
Tokens:   ["gene", "therapy", "studies", "examine", "gene", "expression"]
Counts:   gene: 2, therapy: 1, studies: 1, examine: 1, expression: 1
```

The implemented tokenizer normalizes Unicode text, uses case folding to remove case differences, and standardizes curly apostrophes and common dash characters. It extracts sequences of letters and numbers, preserving internal hyphens and apostrophes, such as `t-cell` and `patient's`. Other punctuation separates tokens. It does not remove common words or reduce words to their roots, so `gene` and `genes` remain different terms.

Once tokenized, text is treated as a **bag of words**: both retrievers use document term counts and ignore word order. They rely on exact matches between normalized terms. For queries, TF-IDF uses term counts, while this BM25 implementation uses each distinct term once.

The shared **inverted index** maps each term to the documents containing it and its count in each document. It also stores document lengths and the total number of documents. At search time, this lets us visit documents containing query terms without scanning every document's text.

## Representing queries and documents as vectors

The **vocabulary** is the set of distinct terms in the corpus. Imagine giving every vocabulary term a fixed position in a vector. We can represent every query and document using those same positions, initially recording a count for each term and zero for an absent term.

For a small example, suppose the entire vocabulary is:

```text
Position:        1        2        3
Term:           gene   therapy   cell

Query counts:    [1,       1,       0]   "gene therapy"
Document counts: [2,       1,       1]   "gene gene therapy cell"
```

These are count vectors. **Both TF-IDF and BM25 transform document term counts into weights** that reflect a term's frequency in the document and its rarity across the corpus. BM25 also adjusts each weight using the document's token count relative to the corpus average.

The methods differ in how they represent the query and combine the weights. Our TF-IDF retriever weights both query and document terms, then compares the vectors using cosine similarity. Our BM25 retriever treats the query as a set of distinct terms and sums their document-term weights. In vector form, that BM25 query has a 1 for each query term in the vocabulary and a 0 elsewhere.

## Cosine similarity in the TF-IDF retriever

TF-IDF defines term weights. Our TF-IDF retriever uses **cosine similarity** to turn those weights into a document score by comparing the query and document vectors. Cosine similarity is a general vector comparison method; here it is the scoring choice for TF-IDF:

$$
\mathrm{cosine}(q,d)
= \frac{\mathbf{q}\cdot\mathbf{d}}{\lVert\mathbf{q}\rVert_2\cdot\lVert\mathbf{d}\rVert_2}
= \frac{\sum_{t\in V} w(t,q)\cdot w(t,d)}
{\sqrt{\sum_{t\in V} w(t,q)^2}\cdot \sqrt{\sum_{t\in V} w(t,d)^2}}
$$

Here, $V$ is the vocabulary, and $w(t,q)$ and $w(t,d)$ are the weights of term $t$ in the query and document.

The numerator is the **dot product**: multiply the query and document weights for each term, then add the products. Only shared terms contribute. A shared term with a large weight in both vectors contributes more than one with small weights.

The denominator is the product of the vectors' **lengths**, or norms. A vector norm is calculated from its term weights; it is different from a document's length measured in tokens. Dividing by these lengths compares the vectors' directions, so simply scaling all the weights in a document by the same amount does not increase its similarity. Each document's norm includes all its weighted terms, including those absent from the query.

With the nonnegative TF-IDF weights used here, cosine similarity ranges from 0 to 1 for nonzero vectors. A score of 0 means no overlap with positive weight; a score of 1 means the vectors point in the same direction. A score is a measure of similarity, not a probability of relevance. If the query has zero norm, the implementation returns no results; documents with zero norm or a zero dot product are omitted.

## TF-IDF: choosing the term weights

**Both retrievers use term frequency and inverse document frequency:** repeated occurrences can indicate a document's subject, and rarer terms help distinguish documents. **Term frequency–inverse document frequency (TF-IDF)** expresses these ideas by multiplying a term-frequency factor by an inverse-document-frequency factor. BM25 uses the same broad structure with different factors.

Our TF-IDF implementation uses the following weight for a term $t$ in a query or document $x$:

$$
w(t,x)=
\begin{cases}
\bigl(1+\ln(\mathrm{tf}(t,x))\bigr)\cdot \ln\!\left(\dfrac{N}{\mathrm{df}(t)}\right),
& \mathrm{tf}(t,x)>0\text{ and }\mathrm{df}(t)>0,\\
0, & \text{otherwise.}
\end{cases}
$$

The quantities are:

- $\mathrm{tf}(t,x)$: the number of occurrences of term $t$ in text $x$.
- $N$: the total number of documents in the corpus.
- $\mathrm{df}(t)$: the number of corpus documents containing $t$, counted once per document.
- $\ln$: the natural logarithm, used throughout both retrievers.

The **term-frequency factor**, $1+\ln(\mathrm{tf})$, rewards repetition with diminishing returns. One occurrence gives a factor of 1, two give about 1.69, and three give about 2.10. Repeating a term three times therefore does not triple its weight.

The **inverse-document-frequency factor**, $\ln(N/\mathrm{df})$, gives rarer terms more weight. A term in every document receives an IDF of 0. A term in 10 of 100 documents receives $\ln(10)\approx2.30$.

For example, if that term occurs three times in a document, its weight is:

$$
(1+\ln 3)\cdot \ln(100/10)\approx4.83.
$$

For TF-IDF, we apply this same weighting formula to the query, using the **corpus document frequencies** for its IDF values. Query terms absent from the corpus receive zero weight. Finally, cosine similarity combines the query and document weights into a ranking score. Both retrievers ignore query terms absent from the corpus, since those terms cannot match any document.

## BM25: choosing term weights and summing them

**BM25** computes a weight for each matching document term using term frequency, inverse document frequency, and document length. Its frequency factor saturates, and its length adjustment uses token counts relative to the corpus average. TF-IDF with cosine also normalizes scores, using weighted vector norms instead.

Our BM25 retriever sums the document-term weights for the distinct query terms. Each summand in the formula below is one term's weight and its contribution to the final score; no cosine normalization is applied:

$$
\mathrm{BM25}(q,d)
=\sum_{t\in\mathrm{unique}(q)}
\mathrm{IDF}_{\mathrm{BM25}}(t)
\frac{\mathrm{tf}(t,d)(k_1+1)}
{\mathrm{tf}(t,d)+k_1\left(1-b+b\dfrac{|d|}{\mathrm{avgdl}}\right)}
$$

The BM25 IDF variant used here is:

$$
\mathrm{IDF}_{\mathrm{BM25}}(t)
=\ln\!\left(1+\frac{N-\mathrm{df}(t)+0.5}{\mathrm{df}(t)+0.5}\right).
$$

As with TF-IDF, rarer terms receive more weight. This variant stays positive even for terms appearing in every document.

The additional quantities are:

- $|d|$: the document's token count, including repetitions.
- $\mathrm{avgdl}$: the average token count across all corpus documents.
- $k_1$: the term-frequency saturation parameter; the default is **1.2**.
- $b$: the document-length normalization parameter; the default is **0.75**.

**Frequency saturation.** Repeated occurrences increase a term's contribution, but the frequency factor approaches a ceiling of $k_1+1$. Larger values of $k_1$ allow repetition to have a stronger effect before saturation. When $k_1=0$, a matching term's contribution is just its IDF, regardless of its count. Terms absent from a document contribute zero and are skipped in the implementation.

**Document-length normalization.** The factor $1-b+b|d|/\mathrm{avgdl}$ is 1 for an average-length document. For the same term count, a longer document receives a smaller contribution and a shorter document receives a larger one when length normalization is active. Setting $b=0$ disables this adjustment; setting $b=1$ applies the full length-ratio adjustment.

For example, consider a term appearing in 10 of 100 documents. In an average-length document where it occurs three times, using the default parameters:

$$
\mathrm{IDF}_{\mathrm{BM25}}(t)
=\ln\!\left(1+\frac{90.5}{10.5}\right)\approx2.26
$$

$$
\text{term contribution}
=\mathrm{IDF}_{\mathrm{BM25}}(t)\cdot \frac{3(1.2+1)}{3+1.2}
\approx3.56.
$$

The final score is the sum of these contributions across distinct query terms. Repeating a query term does not change its BM25 contribution in this implementation. Using the vector representation introduced earlier, this is a dot product: a query weight of 1 includes the corresponding BM25 document-term weight, and a query weight of 0 excludes it. The dot product itself is the score, with no division by vector norms.

## Running the implementation

From the project root, with `uv` installed (Python 3.12+ required), set up the environment:

```bash
uv sync --locked
```

Evaluate both retrievers on the train and test splits, reporting Recall@10, MRR@10, and nDCG@10:

```bash
uv run simple-document-ranking-evaluation
```

Achieved results:

```text
train: 809 queries
Retriever     Recall@10     MRR@10    nDCG@10
TF-IDF           0.7560     0.5841     0.6203
BM25             0.7604     0.6071     0.6397

test: 300 queries
Retriever     Recall@10     MRR@10    nDCG@10
TF-IDF           0.7702     0.5672     0.6100
BM25             0.7655     0.6060     0.6419
```

Search SciFact query `0` and show the top five results from each retriever:

```bash
uv run simple-document-ranking --query-id 0 --top-k 5
```

SciFact downloads automatically on the first run. Change `--query-id` to search another dataset query; add `--rebuild` after changing the corpus or tokenizer.

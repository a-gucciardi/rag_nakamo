1. The RAGAgent only retrieves (and optionally reranks) documents; it does NOT generate an answer.
2. The guard should run after (a) the user’s original question and (b) a candidate model response (produced elsewhere, response agent).
3. The classifier produces labels; the guard maps labels to a decision: allow | block | sanitize.
4. Optionally include retrieved context (top documents) to help the classifier.
5.  enforce the decision before sending anything back to the user.


# Policy:
**1. Harmful user prompt -> block**
**2. Harmful model response AND not a refusal -> sanitize**
**3. Refusal or unharmful -> allow**
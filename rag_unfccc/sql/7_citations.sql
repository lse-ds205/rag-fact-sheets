CREATE TABLE citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    cited_chunk_id UUID NOT NULL REFERENCES doc_chunks(id) ON DELETE CASCADE,
    cited_in_answer_id UUID NOT NULL REFERENCES questions_answers(id) ON DELETE CASCADE
);
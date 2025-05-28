CREATE TABLE questions_answers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    country TEXT NOT NULL REFERENCES countries(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ DEFAULT NOW(),

    question INTEGER NOT NULL REFERENCES questions(id) ON DELETE CASCADE,

    summary TEXT,
    detailed_response TEXT,
    citations UUID[]
);
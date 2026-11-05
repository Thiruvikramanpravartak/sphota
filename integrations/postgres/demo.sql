-- Demo table: texts in several languages with their Sphota s1 codes.
CREATE TABLE IF NOT EXISTS docs (
  id bigserial PRIMARY KEY,
  body text NOT NULL,
  lang text,
  code sphota_code NOT NULL,
  l1 text GENERATED ALWAYS AS (sphota_prefix(code, 1)) STORED
);
CREATE INDEX IF NOT EXISTS docs_code_prefix ON docs (code text_pattern_ops);
CREATE INDEX IF NOT EXISTS docs_l1 ON docs (l1);

INSERT INTO docs (body, lang, code) VALUES
  ('The CEO resigned', 'en', 'JYZA-7M-X0-A8'),
  ('सीईओ ने इस्तीफा दे दिया', 'hi', 'H0VQ-ER-X0-F5'),
  ('தலைமை நிர்வாக அதிகாரி ராஜினாமா செய்தார்', 'ta', 'H0VQ-N7-6Q-JK'),
  ('El director ejecutivo dimitió', 'es', 'JYZA-7M-0Q-BS'),
  ('Le PDG a démissionné', 'fr', 'K7KM-DF-6Q-8D'),
  ('Der Vorstandschef ist zurückgetreten', 'de', 'JYZA-7M-0Z-F5'),
  ('首席执行官辞职了', 'zh', 'JYZA-N7-0Q-BS'),
  ('CEOが辞任した', 'ja', 'JYZA-7M-X0-A8'),
  ('Heavy rain caused flooding in the city center.', 'en', 'S0VF-7M-MT-C1'),
  ('भारी बारिश से शहर के केंद्र में बाढ़ आ गई।', 'hi', '2B51-N7-XK-C1'),
  ('How do I reset my password?', 'en', '5KJE-7M-1F-8Q'),
  ('¿Cómo restablezco mi contraseña?', 'es', '5KJE-7M-1F-PZ'),
  ('The striker scored twice in the final minutes of the match.', 'en', '4PS2-AT-KQ-WA'),
  ('Photosynthesis converts light energy into chemical energy.', 'en', 'G39P-AK-0Z-1K'),
  ('التمثيل الضوئي يحول الطاقة الضوئية إلى طاقة كيميائية.', 'ar', 'CHSK-ER-0Z-8D'),
  ('The central bank raised interest rates by half a percentage point.', 'en', '7PSB-FC-KM-CB'),
  ('Центральный банк повысил процентные ставки на полпроцента.', 'ru', 'WJX1-FC-KW-CB'),
  ('A new species of frog was discovered in the Amazon rainforest.', 'en', 'SH48-AK-6C-8N'),
  ('Uma nova espécie de sapo foi descoberta na floresta amazônica.', 'pt', 'B7CP-AK-YY-A8'),
  ('Add two cups of flour and stir until smooth.', 'en', 'ZN46-N7-ZH-A4'),
  ('দুই কাপ ময়দা যোগ করুন এবং মসৃণ হওয়া পর্যন্ত নাড়ুন।', 'bn', 'ZN46-N7-QP-WA'),
  ('The museum exhibits paintings from the Renaissance.', 'en', 'BNNS-JF-0T-WA'),
  ('I love you', 'en', '7DXV-N7-GY-KD'),
  ('Quantum computers use qubits.', 'en', 'VYMW-88-QK-WA');

-- Try:
--   SELECT body, code FROM docs WHERE code LIKE sphota_like('JYZA-7M');
--   SELECT l1, count(*) FROM docs GROUP BY l1 ORDER BY 2 DESC;
--   SELECT body, sphota_similarity(code, 'JYZA-7M-X0-A8') s FROM docs ORDER BY s DESC LIMIT 5;

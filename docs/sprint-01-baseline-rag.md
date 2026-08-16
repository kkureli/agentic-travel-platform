# Sprint 01 — Baseline RAG

## 1. Sprint Goal

Bu sprintin amacı çalışan bir **baseline / naive RAG pipeline** kurmak ve RAG’in temel parçalarının nasıl çalıştığını anlamaktı.

Sprint sonunda sistem şu akışı gerçekleştirebilir hale geldi:

```text
                 INGESTION

Markdown Documents
        ↓
Document Loader
        ↓
LangChain Documents
        ↓
Recursive Chunking
        ↓
Embedding Model
        ↓
Chunk Embeddings
        ↓
Qdrant Vector Database


                 RETRIEVAL + GENERATION

User Question
        ↓
Query Embedding
        ↓
Qdrant Semantic Search
        ↓
Top-K Chunks
        ↓
LLM Context
        ↓
Generated Answer
        ↓
Sources
```

---

## 2. Project Foundation

Projeyi ileride RAG, agents, GraphRAG, SQL tools, web tools ve LangGraph workflow’ları eklenebilecek şekilde katmanlara ayırdık.

Temel proje yapısı:

```text
app/
├── api/
├── agents/
├── core/
├── knowledge_graph/
├── rag/
│   ├── chunking/
│   ├── embeddings/
│   ├── loaders/
│   └── vector_store/
├── schemas/
├── services/
├── tools/
└── workflows/

data/
├── raw/
└── processed/

tests/
```

Önemli mimari ayrımlar:

```text
API ≠ business logic
RAG ≠ Agent
Agent ≠ Tool
LangGraph workflow ≠ Knowledge Graph
```

---

## 3. FastAPI

FastAPI uygulamanın HTTP API katmanı olarak seçildi.

Temel request akışı:

```text
Client
↓
Uvicorn
↓
FastAPI
↓
Router
↓
Endpoint
↓
Response
```

### Uvicorn

FastAPI bir ASGI framework’tür; Uvicorn uygulamayı çalıştıran ASGI server’dır.

Örnek:

```bash
uvicorn app.main:app --reload
```

### Router

Endpoint’leri doğrudan `main.py` içine doldurmak yerine `APIRouter` ile ayrı modüllere ayırıyoruz.

Örnek:

```text
GET /health
```

Health endpoint ileride container veya deployment ortamında uygulamanın ayakta olup olmadığını kontrol etmek için kullanılabilir.

---

## 4. Configuration Management

API key, database URL veya servis adresleri kod içine hard-code edilmedi.

Akış:

```text
.env
↓
Pydantic Settings
↓
app/core/config.py
↓
Application
```

Örnek ayarlar:

```text
DATABASE_URL
QDRANT_URL
REDIS_URL
NEO4J_URI
OPENAI_API_KEY
```

### Neden?

- Secret’ları source code’dan ayırmak
- Development / staging / production ortamlarını ayırmak
- Typed configuration kullanmak
- Merkezi config yönetimi sağlamak

---

## 5. Infrastructure

Docker Compose ile dört veri servisi ayağa kaldırıldı:

### PostgreSQL

Structured ve relational data için.

Örnek:

```text
customers
bookings
hotels
payments
campaigns
```

### Qdrant

Embedding vector’larını ve RAG chunk’larını saklayan vector database.

Bir Qdrant point kabaca:

```text
Point
├── id
├── vector
└── payload
```

### Redis

İleride caching, agent state, temporary sessions ve tool result cache gibi işler için kullanılacak.

### Neo4j

GraphRAG sprintinde entity ve relationship verilerini saklamak için kullanılacak.

---

## 6. Structured vs Unstructured Data

Structured data tablo mantığına uygunsa PostgreSQL gibi relational DB’ye gider.

Unstructured text ise RAG knowledge base’e uygundur.

Örnek unstructured kaynaklar:

- cancellation policies
- booking policies
- travel insurance
- destination guides
- loyalty program documentation

Akış:

```text
Document
↓
Chunk
↓
Embedding
↓
Vector DB
```

---

## 7. Knowledge Base

İlk baseline için Markdown dokümanları oluşturuldu:

```text
hotel_cancellation_policy.md
hotel_booking_policy.md
travel_insurance.md
antalya_guide.md
loyalty_program.md
```

Markdown kullanılmasının nedenleri:

- sade text formatı
- heading / section yapısını koruması
- loader ve chunking mantığını öğrenmeyi kolaylaştırması

Synthetic data kullanmamızın avantajı, ground truth bilgisini kontrol edebilmemizdir.

---

## 8. Document Loader

Document loader’ın amacı ham dosyaları ortak bir document representation’a dönüştürmektir.

```text
Markdown file
↓
Loader
↓
Document
├── page_content
└── metadata
```

Final pipeline’da LangChain `DirectoryLoader + TextLoader` kullanıldı.

Metadata ileride:

- filtering
- source tracking
- citation
- debugging
- tenant filtering

için kullanılabilir.

---

## 9. Chunking

Büyük dokümanları tek embedding halinde saklamak yerine daha küçük parçalara ayırıyoruz.

Ana sebep storage değildir. Asıl problem, büyük bir dokümanı tek vektöre sıkıştırınca spesifik bilgilerin semantic representation içinde seyrelip retrieval kalitesinin düşebilmesidir.

Baseline olarak `RecursiveCharacterTextSplitter` kullanıldı.

### Recursive Chunking

Recursive splitter mümkün olduğunca doğal text sınırlarını korumaya çalışır:

```text
paragraph
↓
line
↓
space
↓
character
```

### chunk_size

Bir chunk için hedef maksimum uzunluk.

Bu değer evrensel değildir; ileride eval ile seçilecektir.

### chunk_overlap

Komşu chunk’lar arasında ortak içerik bırakır ve chunk sınırında context kaybını azaltır.

Fazla overlap:

- duplicate content
- daha fazla embedding
- daha fazla storage
- benzer retrieval sonuçları

oluşturabilir.

Bir Document chunk’lara bölündüğünde metadata child chunk’lara aktarılır.

---

## 10. Embeddings

Embedding, bir text’i semantic anlamını temsil eden numerical vector’a dönüştürür.

```text
Text
↓
Tokenizer
↓
Token IDs
↓
Embedding Layer
↓
Initial Token Vectors
↓
Transformer
↓
Context-aware Token Representations
↓
Pooling
↓
Final Text / Chunk Embedding
```

### Tokenizer

Metni token ve token ID’lere dönüştürür.

### Transformer

Token representation’larını context’e göre dönüştürür; token’ın kendisini değiştirmez.

### Pooling

Transformer sonrası token başına ayrı vector vardır. Bir chunk için tek vector istediğimiz için bunlar pooling ile tek embedding’e dönüştürülür.

### Baseline Model

`sentence-transformers/all-MiniLM-L6-v2`

Bu model 384-dimensional embedding üretir.

### Aynı model neden gerekli?

Documents ve query aynı vector space içinde olmalıdır:

```text
Documents → Model A → vectors
Query     → Model A → vector
```

---

## 11. Qdrant Indexing

Her chunk Qdrant’a point olarak yazıldı.

```text
chunk.page_content → payload.text
chunk.metadata     → payload.metadata
embedding          → vector
```

Baseline collection:

```text
travel_knowledge
```

Vector size:

```text
384
```

Distance:

```text
Cosine
```

### Upsert

ID yoksa insert, varsa mevcut point update edilir.

### Öğrenilen Bug

`points.append(point)` yanlışlıkla `for` loop’un dışında kaldığında sadece son point Qdrant’a yazılmıştı.

Yanlış:

```python
for ...:
    point = ...

points.append(point)
```

Doğru:

```python
for ...:
    point = ...
    points.append(point)
```

Bu hata, log’da 10 chunk işlendiği görünmesine rağmen Qdrant dashboard’da tek point görünmesiyle fark edildi.

---

## 12. Ingestion Pipeline

Ingestion, ham bilgiyi searchable knowledge base’e dönüştüren offline/indexing sürecidir.

```text
Markdown
↓
Load
↓
Documents
↓
Chunk
↓
Embeddings
↓
Qdrant
```

Kısaca:

```text
Load → Transform → Embed → Store
```

### Ingestion ≠ Retrieval

```text
Ingestion:
Documents → Vector DB

Retrieval:
Question → Vector DB → Relevant Chunks
```

---

## 13. Semantic Retrieval

Kullanıcı query’si de documents ile aynı embedding modeliyle vector’a dönüştürülür:

```text
User Query
↓
Embedding Model
↓
Query Vector
↓
Qdrant
↓
Similarity Search
↓
Top-K Chunks
```

### Top-K

`limit=3` en yakın üç sonucu getir demektir.

Top-K çok yüksek seçilirse:

- noise artabilir
- token maliyeti artabilir
- alakasız bilgi gelebilir
- önemli bilgi context içinde kaybolabilir

### Semantic Search

Query ile document aynı kelimeleri içermek zorunda değildir.

Örneğin:

```text
"cancel my booking"
```

ile:

```text
"hotel cancellation policy"
```

semantic olarak yakın olabilir.

---

## 14. Baseline RAG

Retrieval tamamlandıktan sonra retrieve edilen chunk’lar LLM’e context olarak gönderilir.

```text
Question
↓
Query Embedding
↓
Qdrant
↓
Top-K Context
↓
LLM
↓
Answer
```

Önemli nokta:

> LLM vector database’i kendi başına aramaz; application retrieval yapar ve bulunan context’i modele verir.

RAG’de düşük temperature daha deterministic cevap için tercih edilebilir, ancak `temperature=0` hallucination’ı garanti olarak engellemez.

---

## 15. Retrieval vs Generation Evaluation

### Retrieval Quality

Doğru context’i bulduk mu?

Yanlış/alakasız document gelirse:

```text
Retrieval Failure
```

### Generation Quality

Doğru context verilmesine rağmen model doğru cevap üretti mi?

### Faithfulness Failure

Generated answer retrieved evidence’a sadık değilse faithfulness problemi vardır.

Örnek:

```text
Context:
48 hours before check-in

Answer:
24 hours before check-in
```

Faithfulness failure nedenleri arasında:

- noisy context
- conflicting chunks
- weak prompting
- ambiguous source text
- excessive Top-K
- model interpretation errors

yer alabilir.

Yanlış cevap gördüğümüzde önce retrieval ve generation katmanlarını ayrı incelemek gerekir.

---

## 16. Source Tracking / Citation

Source bilgisini LLM’e yeniden ürettirmek yerine Qdrant metadata’dan alıyoruz.

```text
Qdrant metadata
↓
Application
↓
Response sources
```

Böylece source hallucination riskini azaltıyoruz.

Source, bilginin geldiği dokümandır; citation ise belirli bir claim’in hangi source’a dayandığını gösterme biçimidir.

Sprint 1’de basit source tracking kullanıldı.

---

## 17. Current Baseline Architecture

```text
                 OFFLINE

Knowledge Base
      ↓
Loader
      ↓
Documents
      ↓
Recursive Chunking
      ↓
Sentence Transformer
      ↓
Embeddings
      ↓
Qdrant


                 ONLINE

User Query
     ↓
Same Embedding Model
     ↓
Query Embedding
     ↓
Qdrant Semantic Search
     ↓
Top-K Chunks
     ↓
LLM Context
     ↓
Generation
     ↓
Answer + Sources
```

---

## 18. Why This Is Still Baseline RAG

Şu anda henüz:

- semantic chunking
- metadata filtering
- BM25 / sparse retrieval
- hybrid search
- Reciprocal Rank Fusion
- reranking
- query rewriting
- multi-query retrieval
- contextual retrieval
- retrieval evaluation
- agentic decision making

yok.

Bu nedenle sistem çalışan bir RAG olsa da hâlâ baseline RAG’dir.

---

## 19. Key Engineering Lessons

1. Chunking’in amacı sadece text’i küçültmek değil, retrieval granularity’yi iyileştirmektir.
2. Embedding vector’ının anlamı tek tek sayılarda değil vector space içindeki konumundadır.
3. Query ve documents aynı embedding space içinde olmalıdır.
4. Vector DB’de text ve metadata payload olarak tutulabilir.
5. Retrieval ve generation ayrı ayrı evaluate edilmelidir.
6. Yanlış cevap her zaman LLM hatası değildir.
7. Source tracking mümkün olduğunca deterministic application data’dan gelmelidir.
8. Configuration, storage ve AI logic birbirinden ayrılmalıdır.
9. Pipeline log’ları actual database state ile doğrulanmalıdır.

---

## 20. Interview Summary

> I built a baseline RAG ingestion and retrieval pipeline. Markdown documents are loaded with LangChain, recursively chunked, embedded with a Sentence Transformer model, and indexed in Qdrant together with their metadata. User queries are embedded using the same model, semantic similarity search retrieves the most relevant chunks, and those chunks are provided to an LLM as grounded context. I also keep source metadata in the response so retrieval and generation failures can be debugged separately.

---

## 21. Knowledge Check

Sprint 1 sonunda cevaplayabilmen gereken sorular:

1. Ingestion ile retrieval arasındaki fark nedir?
2. Neden document’ları chunk’lara bölüyoruz?
3. Recursive chunking nasıl çalışır?
4. Chunk overlap neden kullanılır?
5. Tokenizer ile embedding modelinin farkı nedir?
6. Transformer token representation’larını nasıl etkiler?
7. Pooling neden gerekir?
8. Embedding dimension ne demektir?
9. Query ve documents neden aynı embedding modelini kullanmalıdır?
10. Qdrant point hangi parçalardan oluşur?
11. Vector ile payload arasındaki fark nedir?
12. Upsert nedir?
13. Semantic search lexical search’ten nasıl farklıdır?
14. Top-K neden çok yüksek tutulmamalıdır?
15. Retrieval failure nedir?
16. Generation failure nedir?
17. Faithfulness failure nedir?
18. Source metadata neden önemlidir?
19. Citation bilgisini neden LLM’e ürettirmiyoruz?
20. Bu sistem neden henüz advanced RAG değildir?

---

## 22. Next Sprint

### Sprint 02 — Advanced Retrieval & Evaluation

Bir sonraki sprintte:

- structure-aware / semantic chunking
- metadata filtering
- BM25 / sparse retrieval
- hybrid search
- Reciprocal Rank Fusion
- cross-encoder reranking
- query rewriting
- multi-query retrieval
- contextual retrieval
- retrieval evaluation
- LangSmith tracing

işlenecek.

Amaç sadece daha fazla teknik eklemek değil; her yöntemin hangi problemi çözdüğünü anlamak ve retrieval eval sonuçlarına göre gerçekten faydalı olup olmadığına karar vermek.

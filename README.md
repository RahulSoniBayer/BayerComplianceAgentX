# Bayer Compliance Agent

A secure, scalable **AI-assisted document automation platform** that automatically fills Word templates using retrieved content from reference PDFs and AI-generated text.

## 🚀 Features

- **PDF Processing**: Upload reference PDFs, extract text/tables/images, and create searchable embeddings
- **Template Filling**: Upload Word templates with placeholders, automatically fill with AI-generated content
- **Real-time Progress**: WebSocket-based real-time updates during processing
- **Multi-vector DB Support**: Pinecone, Weaviate, or FAISS for vector storage
- **LLM Integration**: MyGenAssist API integration for content generation
- **Batch Processing**: Process multiple templates concurrently
- **Modern UI**: React + Tailwind CSS with real-time notifications

## 🏗️ Architecture

### Backend (Python + FastAPI)
- **PDF Parser**: PyMuPDF for high-quality text and layout extraction
- **Embedding Service**: OpenAI embeddings with pluggable vector databases
- **LLM Service**: MyGenAssist API integration for content generation
- **Template Filler**: Orchestrates the complete automation workflow
- **WebSocket System**: Real-time progress updates and notifications
- **Database**: SQLAlchemy with PostgreSQL/SQLite support

### Frontend (React + Tailwind)
- **Upload Components**: Drag-and-drop file uploads with validation
- **Progress Tracking**: Real-time progress bars and file status updates
- **WebSocket Integration**: Live updates during processing
- **Toast Notifications**: User-friendly success/error messages
- **Responsive Design**: Modern, mobile-friendly interface

## 📋 Requirements

### System Requirements
- Python 3.11+
- Node.js 18+
- PostgreSQL (optional, SQLite supported)

### API Keys Required
- MyGenAssist API key
- OpenAI API key (for embeddings)
- Pinecone API key (if using Pinecone)
- Weaviate credentials (if using Weaviate)

## 🛠️ Installation

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd BayerComplainceAgentXC
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize database**
   ```bash
   # Database tables will be created automatically on first run
   ```

6. **Start the backend**
   ```bash
   cd backend
   python main.py
   # Or with uvicorn:
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm start
   ```

4. **Build for production**
   ```bash
   npm run build
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Database Configuration
DATABASE_URL=sqlite:///./app.db
# DATABASE_URL=postgresql://user:password@localhost/dbname

# Vector Database Configuration
VECTOR_DB_TYPE=pinecone  # Options: pinecone, weaviate, faiss
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=bayer-compliance-docs

WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_weaviate_api_key

# AI/LLM Configuration
MYGENASSIST_API_KEY=your_mygenassist_api_key
MYGENASSIST_BASE_URL=https://api.mygenassist.com/v1
OPENAI_API_KEY=your_openai_api_key_for_embeddings

# Security
SECRET_KEY=your-secret-key-here-change-in-production
ENCRYPTION_KEY=your-32-byte-encryption-key-here

# Application Settings
MAX_FILE_SIZE_MB=50
ALLOWED_FILE_TYPES=pdf,docx
UPLOAD_DIR=./uploads
GENERATED_DIR=./generated

# WebSocket Configuration
WS_HEARTBEAT_INTERVAL=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### Vector Database Options

#### Option 1: Pinecone (Recommended for Production)
```env
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=bayer-compliance-docs
```

#### Option 2: Weaviate
```env
VECTOR_DB_TYPE=weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_api_key
```

#### Option 3: FAISS (Local)
```env
VECTOR_DB_TYPE=faiss
# No additional configuration needed
```

## 🚀 Usage

### 1. Upload Reference PDFs
1. Navigate to the "Reference PDFs" tab
2. Drag and drop PDF files or click "Choose Files"
3. Files are automatically processed and embedded for retrieval

### 2. Process Templates
1. Switch to the "Template Processing" tab
2. Upload Word template files (.docx) with placeholders
3. Add optional context and process flow images
4. Click "Generate Documents" to start processing

### 3. Monitor Progress
- Real-time progress updates via WebSocket
- Toast notifications for file completions
- Download individual files as they complete
- Download all files as a ZIP when complete

### 4. Template Placeholder Format

Use any of these placeholder formats in your Word templates:

```
{{placeholder_text}}
[[placeholder_text]]
<placeholder_text>
%placeholder_text%
```

The system will automatically:
- Extract placeholder text and context type
- Retrieve relevant chunks from PDF embeddings
- Generate appropriate content using AI
- Replace placeholders while preserving formatting

## 🔧 API Endpoints

### PDF Management
- `POST /api/pdf/upload` - Upload PDF files
- `GET /api/pdf/list` - List uploaded PDFs
- `GET /api/pdf/{id}/status` - Get PDF processing status
- `GET /api/pdf/{id}/chunks` - Get PDF chunks
- `DELETE /api/pdf/{id}` - Delete PDF document

### Template Processing
- `POST /api/template/upload` - Upload template files
- `GET /api/template/task/{task_id}/status` - Get task status
- `GET /api/template/download/{file_id}` - Download single file
- `GET /api/template/download_all/{task_id}` - Download all files as ZIP
- `GET /api/template/tasks` - List processing tasks

### WebSocket
- `WS /api/ws/progress/{task_id}` - Real-time progress updates

### Health & Status
- `GET /health` - Health check
- `GET /api/status` - API status and configuration

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📁 Project Structure

```
BayerComplainceAgentXC/
├── backend/
│   ├── api/
│   │   ├── pdf_routes.py
│   │   ├── template_routes.py
│   │   └── websocket_routes.py
│   ├── services/
│   │   ├── embedding_service.py
│   │   ├── retrieval_service.py
│   │   ├── llm_service.py
│   │   └── template_filler_service.py
│   ├── parsers/
│   │   ├── pdf_parser.py
│   │   └── docx_parser.py
│   ├── models/
│   │   └── db_models.py
│   ├── utils/
│   │   ├── config.py
│   │   └── validators.py
│   └── main.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/
│   │   │   ├── UploadPDF.jsx
│   │   │   ├── UploadTemplate.jsx
│   │   │   ├── FileProgressList.jsx
│   │   │   ├── WebSocketManager.jsx
│   │   │   └── ToastManager.jsx
│   │   ├── pages/
│   │   │   └── Home.jsx
│   │   ├── lib/
│   │   │   └── utils.js
│   │   └── App.js
│   ├── public/
│   └── package.json
├── requirements.txt
├── env.example
└── README.md
```

## 🔒 Security Features

- File type and size validation
- Filename sanitization
- Content validation and sanitization
- Secure API key management
- Error handling and logging
- CORS configuration

## 🚀 Deployment

### Docker Deployment (Recommended)

1. **Create Dockerfile for backend**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY backend/ .
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Create Dockerfile for frontend**
   ```dockerfile
   FROM node:18-alpine as build
   WORKDIR /app
   COPY frontend/package*.json ./
   RUN npm install
   COPY frontend/ .
   RUN npm run build

   FROM nginx:alpine
   COPY --from=build /app/build /usr/share/nginx/html
   COPY nginx.conf /etc/nginx/nginx.conf
   ```

3. **Docker Compose**
   ```yaml
   version: '3.8'
   services:
     backend:
       build: ./backend
       ports:
         - "8000:8000"
       environment:
         - DATABASE_URL=postgresql://user:password@db:5432/bayer_agent
       depends_on:
         - db
     
     frontend:
       build: ./frontend
       ports:
         - "3000:80"
     
     db:
       image: postgres:15
       environment:
         POSTGRES_DB: bayer_agent
         POSTGRES_USER: user
         POSTGRES_PASSWORD: password
   ```

### Manual Deployment

1. **Backend**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Set environment variables
   export DATABASE_URL=postgresql://...
   export MYGENASSIST_API_KEY=...
   
   # Run with production server
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
   ```

2. **Frontend**
   ```bash
   cd frontend
   npm run build
   # Serve build directory with nginx or similar
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs` when running the backend
- Review the logs in `./logs/app.log`

## 🔄 Changelog

### Version 1.0.0
- Initial release
- PDF processing and embedding
- Template filling with AI
- Real-time WebSocket updates
- Multi-vector database support
- Modern React frontend
- Complete API documentation

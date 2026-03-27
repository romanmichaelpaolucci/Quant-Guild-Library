# Quant Guild Library - Book App

The **Book App** is a specialized markdown viewer and interactive library designed for the Quant Guild. It leverages a FastAPI backend for high-performance markdown processing and a Next.js frontend for a premium, responsive reading experience.

## Project Structure

```text
book_app/
├── backend/          # FastAPI server for markdown processing and API
├── frontend/         # Next.js web application (React, Tailwind CSS, Lucide)
└── README.md         # You are here
book/                 # Source markdown files (located at project root)
```

## Prerequisites

- **Python**: 3.10 or higher
- **Node.js**: 18.x or higher
- **Package Manager**: npm or yarn

## Getting Started

### 1. Backend Setup

The backend serves the library content via a REST API. It uses `uv` for dependency management and environment isolation.

```bash
# From the project root (Quant-Guild-Library)
# Initialize the environment and install dependencies
uv sync
```

**Running the Backend:**
```bash
# Run using the configured script
uv run book-app
```
*Alternatively, you can run it via the module path:*
```bash
uv run python -m book_app.backend.main
```
The API will be available at `http://127.0.0.1:8000`.

### 2. Frontend Setup

The frontend provides the interactive user interface, showcasing the library content with modern aesthetics.

```bash
# Navigate to the frontend directory
cd book_app/frontend

# Install dependencies
npm install
```

**Running the Frontend:**
```bash
npm run dev
```
The application will be available at `http://localhost:3000`.

## Key Features

- 📘 **Automated Sidebar**: Navigation links are dynamically generated from `book/index.md`.
- ➗ **LaTeX Rendering**: Beautiful mathematical notation supported via KaTeX and MathJax.
- 💻 **Smart Code Blocks**: Collapsible code sections with syntax highlighting for improved readability.
- 🎨 **Modern UI**: Built with Shadcn/UI and custom CSS for a premium "glassmorphism" feel.
- 📱 **Fully Responsive**: Seamless experience across mobile, tablet, and desktop devices.

## Development

- **Backend API**: The frontend expects the backend to be running on port 8000.
- **Content**: Add or modify `.md` files in the `book/` directory to update the library.

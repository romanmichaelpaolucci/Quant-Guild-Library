import os
import re
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt
from mdit_py_plugins.front_matter import front_matter_plugin
from mdit_py_plugins.footnote import footnote_plugin

app = FastAPI(title="Quant Guild Library Server")

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Project Root (up three levels)
BOOK_DIR = BASE_DIR / "book"

# Initialize Markdown renderer
md = (
    MarkdownIt("gfm-like")
    .use(front_matter_plugin)
    .use(footnote_plugin)
    .enable("table")
)

# Setup markdown

# Mount static files for images/assets
# We mount the entire book directory so that relative paths like "Some_files/image.png" work
app.mount("/api/book", StaticFiles(directory=str(BOOK_DIR)), name="book_assets")

def get_sidebar():
    """Extract links from book/index.md for the sidebar."""
    index_path = BOOK_DIR / "index.md"
    if not index_path.exists():
        return []
    
    content = index_path.read_text()
    # Simple regex to find markdown links [Title](File.md)
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\.md\)", content)
    return [{"title": title, "url": f"/{path}"} for title, path in links]

def render_md(file_path: Path):
    """Render markdown file to HTML and adjust links."""
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    content = file_path.read_text()
    
    # Pre-process links: Change [Title](Other.md) to [Title](/Other%20Path)
    content = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\.md\)",
        lambda m: f"[{m.group(1)}](/{m.group(2).replace(' ', '%20')})",
        content
    )
    
    # Pre-process image paths: Change ![](files/img.png) to ![](/api/book/files/img.png)
    # Using /api/book because we'll proxy or mount it
    content = re.sub(r"!\[([^\]]*)\]\((?!/|http)([^)]+)\)", rf"![\1](/api/book/\2)", content)
    
    html = md.render(content)
    return html, content

@app.get("/api/index")
async def get_index():
    sidebar = get_sidebar()
    return {"sidebar": sidebar}

@app.get("/api/page/{page_name}")
async def get_page(page_name: str):
    # Handle both with and without .md extension in the URL
    if page_name.endswith(".md"):
        page_name = page_name[:-3]
    
    file_path = BOOK_DIR / f"{page_name}.md"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    html_content, raw_markdown = render_md(file_path)
    return {
        "title": page_name.replace('_', ' ').title(),
        "content": html_content,
        "markdown": raw_markdown
    }

# Redirect root to /api/index for debugging or just keep for compatibility
@app.get("/")
async def root():
    return {"message": "Quant Guild Library API"}

def start():
    import uvicorn
    # Use string reference so reload works correctly if needed
    uvicorn.run("book_app.backend.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()

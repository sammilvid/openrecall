"""
app.py — Flask web UI for OpenRecall.

Provides:
  /          — timeline slider to browse captured screenshots
  /search    — semantic search powered by ChromaDB
  /static/   — serves the saved WebP screenshots
"""

from threading import Thread

from flask import Flask, render_template_string, request, send_from_directory
from jinja2 import BaseLoader

from openrecall.config import appdata_folder, screenshots_path
from openrecall.database import create_db, get_all_entries, get_timestamps, search_entries
from openrecall.utils import human_readable_time, timestamp_to_human_readable

app = Flask(__name__)

app.jinja_env.filters["human_readable_time"] = human_readable_time
app.jinja_env.filters["timestamp_to_human_readable"] = timestamp_to_human_readable

base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenRecall</title>
  <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.3.0/font/bootstrap-icons.css">
  <style>
    .slider-container { display: flex; flex-direction: column; align-items: center; padding: 20px; }
    .slider { width: 80%; }
    .slider-value { margin-top: 10px; font-size: 1.2em; }
    .image-container { margin-top: 20px; text-align: center; }
    .image-container img { max-width: 100%; height: auto; }
    .entry-meta { font-size: 0.75em; color: #666; padding: 4px 6px; }
  </style>
</head>
<body>
<nav class="navbar navbar-light bg-light">
  <div class="container">
    <form class="form-inline my-2 my-lg-0 w-100 d-flex" action="/search" method="get">
      <input class="form-control flex-grow-1 mr-sm-2" type="search" name="q"
             placeholder="Search your screen history..." aria-label="Search"
             value="{{ request.args.get('q', '') }}">
      <button class="btn btn-outline-secondary my-2 my-sm-0" type="submit">
        <i class="bi bi-search"></i>
      </button>
    </form>
  </div>
</nav>
{% block content %}{% endblock %}
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.3/dist/umd/popper.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
"""


class StringLoader(BaseLoader):
    def get_source(self, environment, template):
        if template == "base_template":
            return base_template, None, lambda: True
        return None, None, None


app.jinja_env.loader = StringLoader()


@app.route("/")
def timeline():
    entries = get_all_entries()  # [{timestamp, filename, app, title, text}, ...]
    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
{% if entries|length > 0 %}
  <div class="container">
    <div class="slider-container">
      <input type="range" class="slider custom-range" id="discreteSlider"
             min="0" max="{{ entries|length - 1 }}" step="1" value="0">
      <div class="slider-value" id="sliderValue">
        {{ entries[0].timestamp | timestamp_to_human_readable }}
      </div>
      <div class="entry-meta" id="entryMeta">
        {{ entries[0].app }} — {{ entries[0].title }}
      </div>
    </div>
    <div class="image-container">
      <img id="timestampImage" src="/static/{{ entries[0].filename }}" alt="Screenshot">
    </div>
  </div>
  <script>
    const entries = {{ entries | tojson }};
    const slider = document.getElementById('discreteSlider');
    const sliderValue = document.getElementById('sliderValue');
    const entryMeta = document.getElementById('entryMeta');
    const img = document.getElementById('timestampImage');

    slider.addEventListener('input', function () {
      const entry = entries[parseInt(slider.value)];
      sliderValue.textContent = new Date(entry.timestamp * 1000).toLocaleString();
      entryMeta.textContent = entry.app + ' — ' + entry.title;
      img.src = '/static/' + entry.filename;
    });

    // Initialise
    sliderValue.textContent = new Date(entries[0].timestamp * 1000).toLocaleString();
    img.src = '/static/' + entries[0].filename;
  </script>
{% else %}
  <div class="container mt-4">
    <div class="alert alert-info">
      Nothing recorded yet — wait a few seconds, or check that your OpenRouter API key is set.
    </div>
  </div>
{% endif %}
{% endblock %}
""",
        entries=entries,
    )


@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    entries = search_entries(q, n_results=24) if q else []

    return render_template_string(
        """
{% extends "base_template" %}
{% block content %}
<div class="container mt-3">
  {% if not entries %}
    <div class="alert alert-secondary">No results for <strong>{{ q }}</strong>.</div>
  {% else %}
    <p class="text-muted">{{ entries|length }} result(s) for <strong>{{ q }}</strong></p>
    <div class="row">
      {% for entry in entries %}
        <div class="col-md-3 mb-4">
          <div class="card h-100">
            <a href="#" data-toggle="modal" data-target="#modal-{{ loop.index0 }}">
              <img src="/static/{{ entry.filename }}" alt="Screenshot" class="card-img-top">
            </a>
            <div class="card-body p-2">
              <small class="text-muted">
                {{ entry.timestamp | timestamp_to_human_readable }}<br>
                {{ entry.app }}
              </small>
            </div>
          </div>
        </div>

        <!-- Full-screen modal -->
        <div class="modal fade" id="modal-{{ loop.index0 }}" tabindex="-1" role="dialog">
          <div class="modal-dialog modal-xl" role="document"
               style="max-width:none;width:100vw;height:100vh;padding:20px;">
            <div class="modal-content" style="height:calc(100vh - 40px);width:calc(100vw - 40px);">
              <div class="modal-header py-1">
                <small>{{ entry.timestamp | timestamp_to_human_readable }} — {{ entry.title }}</small>
                <button type="button" class="close" data-dismiss="modal">&times;</button>
              </div>
              <div class="modal-body p-0">
                <img src="/static/{{ entry.filename }}" alt="Screenshot"
                     style="width:100%;height:100%;object-fit:contain;">
              </div>
              <div class="modal-footer py-1">
                <small class="text-muted">{{ entry.text[:300] }}{% if entry.text|length > 300 %}…{% endif %}</small>
              </div>
            </div>
          </div>
        </div>
      {% endfor %}
    </div>
  {% endif %}
</div>
{% endblock %}
""",
        entries=entries,
        q=q,
    )


@app.route("/static/<filename>")
def serve_image(filename):
    return send_from_directory(screenshots_path, filename)


if __name__ == "__main__":
    create_db()

    from openrecall.screenshot import record_screenshots_thread
    t = Thread(target=record_screenshots_thread, daemon=True)
    t.start()

    print(f"OpenRecall running at http://localhost:8082")
    print(f"Data folder: {appdata_folder}")
    app.run(port=8082)

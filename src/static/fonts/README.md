Drop any optional custom font files and a matching `custom.css` file into this directory.

This directory is gitignored except for this README so the main UI can reference:

```html
<link rel="stylesheet" href="/static/fonts/custom.css" onerror="this.remove()">
```


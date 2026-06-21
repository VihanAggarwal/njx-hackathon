# Deploy this demo to Vercel

This `web/` folder is a self-contained static site (no server, no API key).

**Easiest:** drag-and-drop this `web` folder onto https://vercel.com/new

**CLI:**
```bash
cd web
npx vercel        # preview
npx vercel --prod # production
```

Regenerate after a new benchmark: `python demo/build_static_site.py`.

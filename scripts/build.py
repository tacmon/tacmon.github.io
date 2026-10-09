"""Build the static blog: python scripts/build.py (requires markdown-it-py)."""
from pathlib import Path
import html
import re
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://tacmon.github.io'
TITLE = '像读竞赛题解一样理解 PPO 与 SAC'
DESC = '从题意、DP 建模到更新公式与伪代码，用程序设计竞赛题解的方式理解 PPO 和 SAC。'
source = (ROOT / 'content/sac-ppo.md').read_text()
equations = []

def protect(match):
    raw = match.group()
    display = raw.startswith('$$')
    tex = raw[2:-2].strip() if display else raw[1:-1]
    tag = 'div' if display else 'span'
    equations.append(f'<{tag} class="math {"display" if display else "inline"}" data-tex="{html.escape(tex, quote=True)}">{html.escape(tex)}</{tag}>')
    return f'MATHPLACEHOLDER{len(equations)-1}END'

protected = re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$', protect, source)
body = MarkdownIt('commonmark', {'html': False}).enable('table').render(protected)
for i, equation in enumerate(equations):
    placeholder = f'MATHPLACEHOLDER{i}END'
    if equation.startswith('<div'):
        body = body.replace(f'<p>{placeholder}</p>', equation)
    else:
        body = body.replace(placeholder, equation)
body = re.sub(r'<h1>.*?</h1>\n', '', body, count=1)
toc = []
def heading(match):
    number = len(toc) + 1
    label = match.group(1)
    toc.append(f'<a href="#section-{number}">{label}</a>')
    return f'<h2 id="section-{number}">{label}</h2>'
body = re.sub(r'<h2>(.*?)</h2>', heading, body)
body = body.replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')

def page(title, description, path, main, math=False):
    math_head = '<link rel="stylesheet" href="/assets/katex/katex.min.css"><script defer src="/assets/katex/katex.min.js"></script><script defer src="/assets/math.js"></script>' if math else ''
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · tacmon</title><meta name="description" content="{html.escape(description, quote=True)}">
<meta name="theme-color" content="#f8f7f3"><link rel="canonical" href="{BASE}{path}">
<meta property="og:title" content="{html.escape(title, quote=True)}"><meta property="og:description" content="{html.escape(description, quote=True)}"><meta property="og:type" content="{'article' if math else 'website'}"><meta property="og:url" content="{BASE}{path}">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="/assets/style.css">{math_head}</head><body>
<a class="skip" href="#main">跳到正文</a><header class="site-header"><a class="brand" href="/">tacmon<span> / 学习笔记</span></a><nav aria-label="主导航"><a href="/">文章</a><a href="https://github.com/tacmon">GitHub ↗</a></nav></header>
{main}
<footer class="site-footer"><span>tacmon · 把问题想清楚，把过程写下来。</span><a href="https://github.com/tacmon/tacmon.github.io">博客源码 ↗</a></footer></body></html>'''

home = '''<main id="main" class="home"><section class="home-intro"><p class="eyebrow">LEARNING IN PUBLIC</p><h1>把问题拆开，<br>把理解留下。</h1><p class="lead">关于算法、强化学习与研究实践的笔记。<br>从问题出发，写下推导，也记录实现。</p></section>
<section class="posts" aria-labelledby="posts-title"><div class="section-label"><h2 id="posts-title">最新文章</h2><span>01 / NOTES</span></div><a class="post-card" href="/posts/sac-ppo/"><div class="post-meta"><span>强化学习</span><span>算法题解</span></div><h2>像读竞赛题解一样<br>理解 PPO 与 SAC <span class="arrow">↗</span></h2><p>从题意与 DP 建模出发，推导策略更新，读懂伪代码，再检查那些最容易 WA 的细节。</p><span class="read-link">阅读全文 →</span></a></section></main>'''
(ROOT / 'index.html').write_text(page('学习笔记', DESC, '/', home))

article = f'''<main id="main"><div class="article-header"><a class="back" href="/">← 全部文章</a><p class="eyebrow">REINFORCEMENT LEARNING / 算法题解</p><h1>{TITLE}</h1><p class="lead">从“当前收益 + 后续价值”开始，理解两种策略学习方法。</p><div class="post-meta"><span>tacmon</span><span>PPO-Clip · 连续动作 SAC</span><a href="/content/sac-ppo.md">Markdown 源文 ↓</a></div></div>
<div class="reading-layout"><aside class="toc" aria-label="文章目录"><p>本文目录</p>{''.join(toc)}<a class="back-top" href="#main">回到顶部 ↑</a></aside><article class="prose">{body}<div class="article-end">— 本文完 —</div></article></div></main>'''
dest = ROOT / 'posts/sac-ppo'
dest.mkdir(parents=True, exist_ok=True)
(dest / 'index.html').write_text(page(TITLE, DESC, '/posts/sac-ppo/', article, math=True))
(ROOT / '.nojekyll').touch()
(ROOT / 'robots.txt').write_text(f'User-agent: *\nAllow: /\nSitemap: {BASE}/sitemap.xml\n')
(ROOT / 'sitemap.xml').write_text(f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>{BASE}/</loc></url><url><loc>{BASE}/posts/sac-ppo/</loc></url></urlset>')
print(f'Built homepage and article: {len(equations)} equations, {len(toc)} sections.')

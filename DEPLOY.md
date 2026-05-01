# 🚀 部署指南

本文档介绍如何把《LLM 百案录》仓库部署成在线可访问的网站。

---

## 方式 A · GitHub Pages（推荐｜零成本｜自动）

### 快速部署（3 分钟）

1. **Fork 本仓库** 到你的 GitHub 账号
2. 在仓库设置中：
   - 进入 `Settings → Pages`
   - **Source**：选择 `GitHub Actions`
3. 推送任何提交到 `main` 分支，工作流会自动触发
4. 等待约 1 分钟，访问：
   ```
   https://<your-username>.github.io/<repo-name>/
   ```

> ✅ 仓库已包含 `.github/workflows/pages.yml`——这是预配置好的 GitHub Actions
> 工作流，会自动把整个仓库（作为 Docsify 静态站点）部署到 GitHub Pages。

### 备用：从分支部署（如果不想用 Actions）
1. `Settings → Pages`
2. **Source**：`Deploy from a branch`
3. **Branch**：`main` / `(root)`
4. 保存即可

---

## 方式 B · Vercel（推荐｜全球 CDN｜自定义域名）

1. 登录 [vercel.com](https://vercel.com)
2. **New Project** → 导入 GitHub 仓库
3. **Framework Preset**：`Other`（其他）
4. **Build Command**：留空
5. **Output Directory**：留空（或填 `.`）
6. **Deploy** 按钮一键部署

部署完后默认域名形如：`llm-papers-casefile.vercel.app`

---

## 方式 C · Netlify

1. 登录 [netlify.com](https://netlify.com)
2. **Add new site → Import an existing project**
3. 选择 GitHub 仓库
4. **Build settings**：均留空（Docsify 是纯静态站点）
5. **Deploy site**

---

## 方式 D · 本地预览（开发环境）

```bash
# 方式 1：用 docsify-cli（推荐）
npx docsify-cli serve .
# 访问 http://localhost:3000

# 方式 2：用 Python
python -m http.server 3000
# 访问 http://localhost:3000

# 方式 3：全局安装
npm install -g docsify-cli
docsify serve .
```

---

## 自定义域名

### 在 GitHub Pages
1. 仓库根目录创建 `CNAME` 文件，内容为你的域名（如 `llm.example.com`）
2. 在 DNS 服务商添加：
   - `A` 记录指向 GitHub Pages IP（`185.199.108.153` 等）
   - 或 `CNAME` 记录指向 `<your-username>.github.io`
3. 等待 DNS 生效（通常 5-30 分钟）

### 在 Vercel / Netlify
平台界面上直接添加自定义域名即可，会自动配 HTTPS。

---

## 常见问题

### Q1：站点能访问，但侧栏 / 数学公式不显示？
检查浏览器控制台是否有 CDN 加载失败。如果你的网络访问 `cdn.jsdelivr.net` 受限，
可以把 `index.html` 中所有 CDN 链接换成 `unpkg.com` 或国内 CDN（如 `cdnjs.cloudflare.com`）。

### Q2：mermaid 流程图不渲染？
确认 `index.html` 中已加载 `mermaid` 和 `docsify-mermaid`。如果还有问题，
可以在浏览器控制台运行 `mermaid.run()` 强制重渲。

### Q3：搜索功能找不到部分笔记？
首次构建索引需 1-2 秒，且 `paths: 'auto'` 模式只索引侧栏中列出的页面。
确保新笔记已加入 `_sidebar.md`。

### Q4：本地预览报 CORS 错误？
不能直接 `file://` 打开 `index.html`，必须通过 HTTP 服务器（`python -m http.server` 即可）。

### Q5：想完全离线使用？
把 `index.html` 中所有 CDN 链接的脚本和 CSS 文件下载到本地 `assets/`，
然后改成相对路径引用即可。

---

## 部署清单（Checklist）

发布前确保：

- [ ] `README.md` 中的 GitHub 仓库链接已替换为你的实际仓库
- [ ] `index.html` 中的 `repo` 字段已更新
- [ ] `_coverpage.md` / `_navbar.md` / `_sidebar.md` 中的 GitHub 链接已更新
- [ ] `LICENSE` 中的 copyright holder 已更新（如有需要）
- [ ] 至少一篇笔记测试过：内部链接、数学公式、mermaid 图都正常
- [ ] 确认 `.nojekyll` 文件存在（避免 GitHub Pages 把下划线开头文件忽略）

---

> 📚 *部署完成后，把网址分享出去——让更多人加入"破案"。*

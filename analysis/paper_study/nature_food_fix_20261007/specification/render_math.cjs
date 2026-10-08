// Deterministic MathJax TeX→SVG/MathML conversion for static scientific documents.
const fs = require('fs');
const path = require('path');
const base = path.join(__dirname, '.conversion_tools/node_modules/mathjax-full/js');
const {mathjax} = require(path.join(base, 'mathjax.js'));
const {TeX} = require(path.join(base, 'input/tex.js'));
const {SVG} = require(path.join(base, 'output/svg.js'));
const {liteAdaptor} = require(path.join(base, 'adaptors/liteAdaptor.js'));
const {RegisterHTMLHandler} = require(path.join(base, 'handlers/html.js'));
const {AllPackages} = require(path.join(base, 'input/tex/AllPackages.js'));
const {SerializedMmlVisitor} = require(path.join(base, 'core/MmlTree/SerializedMmlVisitor.js'));
const {STATE} = require(path.join(base, 'core/MathItem.js'));
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({packages: AllPackages});
const svg = new SVG({fontCache: 'local'});
const doc = mathjax.document('', {InputJax: tex, OutputJax: svg});
const visitor = new SerializedMmlVisitor();
const requests = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const output = requests.map(request => {
  const mml = visitor.visitTree(doc.convert(request.tex, {display: request.display, end: STATE.COMPILED}));
  const result = adaptor.outerHTML(doc.convert(request.tex, {display: request.display}));
  if (mml.includes('<merror') || result.includes('data-mjx-error')) {
    throw new Error(`Math rendering error for ${request.id}: ${request.tex}`);
  }
  const svgText = result.slice(result.indexOf('<svg '), result.lastIndexOf('</svg>') + 6);
  return {...request, mml, svg: svgText};
});
fs.writeFileSync(process.argv[3], JSON.stringify(output));

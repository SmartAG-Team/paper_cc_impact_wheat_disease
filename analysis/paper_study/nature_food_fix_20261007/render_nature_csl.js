// Render numeric Nature-style references in first-citation order.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const root = path.resolve(__dirname, '../../..');
const context = {console};
vm.runInNewContext(fs.readFileSync(path.join(root, 'analysis/paper_study/manuscript/csl_vendor/citeproc.js'), 'utf8'), context);
const CSL = context.CSL;
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const records = Object.fromEntries(input.items.map(item => [item.id, item]));
const system = {
  retrieveItem: id => records[id],
  retrieveLocale: () => fs.readFileSync(path.join(root, 'analysis/paper_study/manuscript/csl_vendor/locales-en-US.xml'), 'utf8')
};
const style = fs.readFileSync(path.join(__dirname, 'references/nature.csl'), 'utf8');
const processor = new CSL.Engine(system, style, 'en-GB');
processor.updateItems(input.ordered_ids);
const citations = {};
for (const [token, ids] of Object.entries(input.clusters)) {
  citations[token] = processor.makeCitationCluster(ids.map(id => ({id})));
}
const bibliography = processor.makeBibliography();
fs.writeFileSync(process.argv[3], JSON.stringify({
  style_id: 'http://www.zotero.org/styles/nature',
  citations,
  bibliography_ids: bibliography[0].entry_ids.map(ids => ids[0]),
  bibliography_html: bibliography[1],
  hanging_indent: bibliography[0].hangingindent,
  processor_version: CSL.PROCESSOR_VERSION
}, null, 2) + '\n');

const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const errs = [];
  for (const scheme of ['light', 'dark']) {
    const p = await b.newPage({ viewport: { width: 1360, height: 860 }, colorScheme: scheme });
    p.on('pageerror', e => errs.push(String(e)));
    p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
    await p.goto('http://127.0.0.1:8799/');
    await p.waitForTimeout(1500);
    for (const s of await p.$$('details.tm > summary')) await s.click();
    await p.screenshot({ path: 'h1_' + scheme + '.png' });
    await p.click('.pr a');
    await p.waitForTimeout(800);
    await p.screenshot({ path: 'h2_' + scheme + '.png' });
    await p.keyboard.press('Escape');
    await p.click('#btnSettings');
    await p.waitForTimeout(400);
    await p.screenshot({ path: 'h3_' + scheme + '.png' });
    await p.close();
  }
  const p = await b.newPage({ viewport: { width: 1024, height: 700 } });
  await p.goto('http://127.0.0.1:8799/');
  await p.waitForTimeout(1200);
  await p.screenshot({ path: 'h4_small.png' });
  const ow = await p.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  console.log('errors:', JSON.stringify(errs), 'overflow:', ow);
  await b.close();
})();

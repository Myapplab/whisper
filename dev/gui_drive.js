const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  for (const scheme of ['dark','light']) {
    const p = await b.newPage({ viewport: { width: 1360, height: 860 }, colorScheme: scheme });
    const errs = []; p.on('pageerror', e => errs.push(String(e))); p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
    await p.goto('http://127.0.0.1:8799/');
    await p.waitForTimeout(1500);
    await p.screenshot({ path: 'g0_' + scheme + '.png' });
    const dir = process.env.STT_TEST_DIR || '.';  // test medyası klasörü (dev/README.md)
    await p.fill('#links', [dir + "/Mikel Arteta's pre-Chelsea press conference.flac",
      dir + "/Frank Lampard Post-Match Press Conference - Everton vs Coventry City.mp4"].join("\n"));
    await p.click('#btnStart');
    await p.waitForTimeout(6000);
    await p.screenshot({ path: 'g1_' + scheme + '.png' });
    await p.waitForFunction(() => document.querySelectorAll('.job .pill.ok, .job .pill.warn').length >= 2, null, { timeout: 180000 });
    await p.waitForTimeout(1500);
    await p.click('#btnLog');
    await p.screenshot({ path: 'g2_' + scheme + '.png' });
    await p.click('#btnLog');
    await p.click('.pr a');
    await p.waitForTimeout(800);
    await p.screenshot({ path: 'g3_' + scheme + '.png' });
    console.log('errors:', JSON.stringify(errs));
  }
  await b.close();
})();

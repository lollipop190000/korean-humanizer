/* Generated installers contain private experiment data. Share responder links only.
 * Standalone Apps Script, V8. No external HTTP calls or API keys.
 */
const SURVEY_PACKAGE = /*__SURVEY_PACKAGE__*/null;
const HEADERS = {
  Studies: ['study_id', 'package_sha256', 'title', 'tasks', 'target_raters', 'status'],
  Forms: ['study_id', 'batch_id', 'form_id', 'status', 'responder_url', 'edit_url', 'items_json', 'error'],
  Tasks: ['study_id', 'batch_id', 'entry_json'],
  Raw: ['form_id', 'response_id', 'received_at', 'response_json'],
  Export: ['study_id', 'package_sha256', 'rating_json'],
  Issues: ['study_id', 'response_id', 'task_id', 'rater_id', 'reason'],
  Dashboard: ['study_id', 'candidate', 'edit_expectation', 'tasks', 'rated_tasks', 'two_raters',
    'disagreements', 'ratings', 'meaning_error_ratings', 'register_drift_ratings',
    'improved', 'same', 'worse', 'unassessable_ratings'],
  Coverage: ['study_id', 'batch_id', 'task_id', 'raters', 'target_raters', 'disagreement', 'status'],
  Feedback: ['study_id', 'response_id', 'rater_id', 'comment']
};
const OPTIONS = {
  naturalness_vs_input: {'좋아짐': 'improved', '비슷함': 'same', '나빠짐': 'worse', '판단 어려움': 'unassessable'},
  propositions: {'유지됨': 'preserved', '빠짐': 'omitted', '뜻이 바뀜': 'changed', '판단 어려움': 'unassessable'},
  unsupported_addition: {'없음': 'absent', '있음': 'present', '판단 어려움': 'unassessable'},
  register_drift: {'없음': 'absent', '있음': 'present', '판단 어려움': 'unassessable'}
};

function workbook_() {
  const props = PropertiesService.getScriptProperties();
  let id = props.getProperty('KH_WORKBOOK_ID');
  if (!id) {
    const ss = SpreadsheetApp.create('Korean Humanizer · 평가 관리');
    id = ss.getId();
    props.setProperty('KH_WORKBOOK_ID', id);
  }
  const ss = SpreadsheetApp.openById(id);
  Object.keys(HEADERS).forEach(name => {
    if (!ss.getSheetByName(name)) {
      const sheet = ss.insertSheet(name);
      sheet.appendRow(HEADERS[name]);
      sheet.setFrozenRows(1);
      sheet.getRange(1, 1, 1, HEADERS[name].length).setFontWeight('bold');
    }
  });
  return ss;
}

function rows_(ss, name) {
  const sheet = ss.getSheetByName(name);
  const values = sheet.getDataRange().getValues();
  return values.slice(1).filter(row => row[0] !== '').map(row => {
    const obj = {};
    HEADERS[name].forEach((key, i) => obj[key] = row[i]);
    return obj;
  });
}

// Treat user-entered spreadsheet values as text, never as formulas.
function safeCell_(value) {
  return typeof value === 'string' && /^[=+@\-\t\r]/.test(value) ? "'" + value : value;
}
function append_(ss, name, values) {
  ss.getSheetByName(name).appendRow(HEADERS[name].map(k => safeCell_(values[k] ?? '')));
}
function replace_(ss, name, values) {
  const sheet = ss.getSheetByName(name);
  if (sheet.getLastRow() > 1) sheet.getRange(2, 1, sheet.getLastRow() - 1, HEADERS[name].length).clearContent();
  if (values.length) sheet.getRange(2, 1, values.length, HEADERS[name].length)
    .setValues(values.map(v => HEADERS[name].map(k => safeCell_(v[k] ?? ''))));
}
function updateForm_(ss, index, form) {
  ss.getSheetByName('Forms').getRange(index + 2, 1, 1, HEADERS.Forms.length)
    .setValues([HEADERS.Forms.map(k => safeCell_(form[k] ?? ''))]);
}

/** Run once per new generated package, in the SAME Apps Script project. */
function setup() {
  requireOwner_();
  if (!SURVEY_PACKAGE || SURVEY_PACKAGE.schema_version !== 1) throw new Error('생성된 Install.gs를 사용하세요.');
  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    const ss = workbook_();
    const p = SURVEY_PACKAGE;
    const studies = rows_(ss, 'Studies');
    const old = studies.find(s => s.study_id === p.study_id);
    if (old && old.package_sha256 !== p.package_sha256) throw new Error('같은 실험 ID의 내용이 바뀌었습니다. 새 ID로 준비하세요.');
    if (!old) append_(ss, 'Studies', {
      study_id: p.study_id, package_sha256: p.package_sha256, title: p.title,
      tasks: p.batches.reduce((n, b) => n + b.entries.length, 0),
      target_raters: p.target_raters_per_task, status: 'active'
    });
    // Registration resumes after a quota/timeout without duplicating task rows.
    const known = new Set(rows_(ss, 'Tasks').map(r => r.study_id + '/' + JSON.parse(r.entry_json).task.task_id));
    const forms = new Set(rows_(ss, 'Forms').map(r => r.study_id + '/' + r.batch_id));
    p.batches.forEach(batch => {
      batch.entries.forEach(entry => {
        const k = p.study_id + '/' + entry.task.task_id;
        if (!known.has(k)) {
          append_(ss, 'Tasks', {study_id: p.study_id, batch_id: batch.id, entry_json: JSON.stringify(entry)});
          known.add(k);
        }
      });
      const k = p.study_id + '/' + batch.id;
      if (!forms.has(k)) {
        append_(ss, 'Forms', {study_id: p.study_id, batch_id: batch.id, status: 'pending'});
        forms.add(k);
      }
    });
    if (!ScriptApp.getProjectTriggers().some(t => t.getHandlerFunction() === 'tick_'))
      ScriptApp.newTrigger('tick_').timeBased().everyMinutes(1).create();
    rebuild_(ss);
    guide_(ss);
    console.log('관리 시트: ' + ss.getUrl());
    console.log('Forms 탭에 설문 링크가 차례로 생성됩니다. 관리 시트·스크립트는 평가자에게 공유하지 마세요.');
  } finally { lock.releaseLock(); }
}

/** One form per tick; once all are ready, refresh responses every five minutes. */
function tick_() {
  const lock = LockService.getScriptLock();
  if (!lock.tryLock(1000)) return;
  try {
    const ss = workbook_();
    const forms = rows_(ss, 'Forms');
    const studies = rows_(ss, 'Studies');
    const active = new Set(studies.filter(s => s.status === 'active').map(s => s.study_id));
    const index = forms.findIndex(f => active.has(f.study_id) && ['pending', 'building'].includes(f.status));
    if (index >= 0) {
      const f = forms[index];
      try { createForm_(ss, f, index, studies.find(s => s.study_id === f.study_id)); }
      catch (error) {
        f.status = 'error'; f.error = String(error); updateForm_(ss, index, f);
        console.error('Forms 탭의 error를 확인하고 해결 후 retryErrors()를 실행하세요.');
      }
    }
    const props = PropertiesService.getScriptProperties();
    if (Date.now() - Number(props.getProperty('KH_LAST_SYNC') || 0) >= 300000) {
      sync_(ss, active);
      rebuild_(ss);
      props.setProperty('KH_LAST_SYNC', String(Date.now()));
    }
  } finally { lock.releaseLock(); }
}

function createForm_(ss, f, index, study) {
  const entries = rows_(ss, 'Tasks').filter(r => r.study_id === f.study_id && r.batch_id === f.batch_id)
    .map(r => JSON.parse(r.entry_json));
  // Store the ID before adding items so interruption cannot create another form on resume.
  let form;
  if (f.form_id) {
    form = FormApp.openById(f.form_id);
    if (form.getResponses().length) throw new Error('응답이 있는 설문은 다시 만들 수 없습니다.');
    form.setPublished(false);
    while (form.getItems().length) form.deleteItem(0);
  } else {
    form = FormApp.create(study.title + ' · ' + f.batch_id, false);
    f.form_id = form.getId();
  }
  f.status = 'building'; f.error = ''; updateForm_(ss, index, f);
  form.setDescription('한국어 원문과 수정문을 비교하는 설문입니다. 총 ' + entries.length + '개입니다. '
    + 'AI인지 맞히는 시험이 아닙니다. 이미 자연스러운 원문은 그대로 유지해도 좋습니다. '
    + '모델 이름은 평가가 끝날 때까지 공개하지 않습니다. 이름·이메일 대신 안내받은 평가자 코드를 사용하세요. '
    + '응답과 의견은 문장 품질 연구·개발에 사용됩니다. 자유 의견에 개인정보를 쓰지 마세요.')
    .setCollectEmail(false).setLimitOneResponsePerUser(false).setAllowResponseEdits(false)
    .setPublishingSummary(false).setProgressBar(true).setShuffleQuestions(false)
    .setShowLinkToRespondAgain(false).setConfirmationMessage('평가를 저장했습니다. 감사합니다. 다음 묶음은 담당자의 안내를 따라 주세요.');
  const mapping = {tasks: [], rater: '', consent: '', feedback: ''};
  const code = form.addTextItem().setTitle('평가자 코드').setHelpText('담당자가 준 코드를 계속 사용하세요. 이름이나 이메일은 적지 마세요.')
    .setRequired(true).setValidation(FormApp.createTextValidation().requireTextMatchesPattern('^[A-Za-z0-9_-]{4,40}$').build());
  mapping.rater = String(code.getId());
  mapping.consent = String(form.addCheckboxItem().setTitle('참여 확인')
    .setChoiceValues(['한국어 글을 직접 읽고 평가하며, 응답의 연구·개발 이용에 동의합니다.']).setRequired(true).getId());
  entries.forEach((entry, n) => {
    const task = entry.task;
    form.addPageBreakItem().setTitle('문장 ' + (n + 1) + ' / ' + entries.length)
      .setHelpText('원문\n' + task.input + '\n\n수정문\n' + (task.output || '(생성된 수정문 없음)')
        + (task.finish_reason === 'stop' ? '' : '\n\n이 수정문은 생성이 정상적으로 끝나지 않았습니다.'));
    const ids = {task_id: task.task_id};
    ids.naturalness_vs_input = String(form.addMultipleChoiceItem().setTitle('원문에 비해 수정문이 얼마나 자연스러운가요?')
      .setHelpText('문장 길이나 변경량 자체를 좋고 나쁨의 기준으로 삼지 마세요.')
      .setChoiceValues(Object.keys(OPTIONS.naturalness_vs_input)).setRequired(true).getId());
    ids.propositions = String(form.addGridItem().setTitle('원문의 다음 내용이 수정문에 유지됐나요?')
      .setRows(task.propositions.map(p => p.id + '. ' + p.claim))
      .setColumns(Object.keys(OPTIONS.propositions)).setRequired(true).getId());
    ids.unsupported_addition = String(form.addMultipleChoiceItem().setTitle('원문에 없던 사실·주장·감정이 추가됐나요?')
      .setChoiceValues(Object.keys(OPTIONS.unsupported_addition)).setRequired(true).getId());
    ids.register_drift = String(form.addMultipleChoiceItem().setTitle('원래 어투·높임말·작성자 관점이 달라졌나요?')
      .setChoiceValues(Object.keys(OPTIONS.register_drift)).setRequired(true).getId());
    ids.evidence = String(form.addParagraphTextItem().setTitle('판단 근거 또는 개선 의견')
      .setHelpText('좋아짐·나빠짐·누락·변경·추가·판단 어려움에는 해당 표현과 이유를 적어 주세요. '
        + '내용 점검은 p1, p2처럼 번호를 붙이면 좋습니다. 모두 유지되고 비슷하면 "특이사항 없음"이라고 적어도 됩니다.')
      .setRequired(true).getId());
    mapping.tasks.push(ids);
  });
  mapping.feedback = String(form.addParagraphTextItem().setTitle('설문 자체에 대한 의견 (선택)')
    .setHelpText('질문이 어렵거나 불편했던 점을 알려 주세요.').getId());
  f.items_json = JSON.stringify(mapping);
  // Save mapping before publishing; after publication the form must never be rebuilt.
  f.status = 'ready'; f.edit_url = form.getEditUrl(); updateForm_(ss, index, f);
  try {
    form.setPublished(true);
    form.setAcceptingResponses(true);
    f.responder_url = form.getPublishedUrl();
    updateForm_(ss, index, f);
  } catch (error) {
    f.status = 'publish_error'; f.error = String(error); updateForm_(ss, index, f);
    throw error;
  }
}

/** Manual refresh; useful immediately after your first real submission. */
function refresh() {
  requireOwner_();
  const lock = LockService.getScriptLock(); lock.waitLock(30000);
  try {
    const ss = workbook_();
    sync_(ss, new Set(rows_(ss, 'Studies').map(s => s.study_id)));
    rebuild_(ss);
    console.log(ss.getUrl());
  } finally { lock.releaseLock(); }
}
function sync_(ss, active) {
  const known = new Set(rows_(ss, 'Raw').map(r => r.form_id + '/' + r.response_id));
  const forms = rows_(ss, 'Forms').filter(f => active.has(f.study_id) && f.form_id && f.items_json);
  const props = PropertiesService.getScriptProperties();
  const cursor = Number(props.getProperty('KH_SYNC_CURSOR') || 0);
  // Limit service calls per invocation; resume round-robin for larger collections.
  for (let offset = 0; offset < Math.min(forms.length, 24); offset++) {
    const f = forms[(cursor + offset) % forms.length];
    FormApp.openById(f.form_id).getResponses().forEach(response => {
      const identity = f.form_id + '/' + response.getId();
      if (known.has(identity)) return;
      const values = {};
      response.getItemResponses().forEach(item => values[String(item.getItem().getId())] = item.getResponse());
      append_(ss, 'Raw', {form_id: f.form_id, response_id: response.getId(),
        received_at: response.getTimestamp().toISOString(), response_json: JSON.stringify(values)});
      known.add(identity);
    });
  }
  props.setProperty('KH_SYNC_CURSOR', String(forms.length ? (cursor + Math.min(24, forms.length)) % forms.length : 0));
}

/** Pure normalization; also tested locally with Node, without contacting Google. */
function normalize_(raw, f, entries) {
  const values = JSON.parse(raw.response_json);
  const mapping = JSON.parse(f.items_json);
  const rater = String(values[mapping.rater] || '').trim();
  if (!/^[A-Za-z0-9_-]{4,40}$/.test(rater)) throw new Error('유효한 평가자 코드가 없습니다.');
  if (!Array.isArray(values[mapping.consent]) || values[mapping.consent].length !== 1
      || values[mapping.consent][0] !== '한국어 글을 직접 읽고 평가하며, 응답의 연구·개발 이용에 동의합니다.')
    throw new Error('참여 확인이 없습니다.');
  const ratings = mapping.tasks.map(ids => {
    const entry = entries.find(e => e.task.task_id === ids.task_id);
    if (!entry) throw new Error('평가 문항 대응이 없습니다.');
    const task = entry.task;
    const grid = values[ids.propositions];
    if (!Array.isArray(grid) || grid.length !== task.propositions.length) throw new Error('내용 점검이 불완전합니다.');
    const evidence = String(values[ids.evidence] || '').trim();
    if (!evidence) throw new Error('판단 근거가 없습니다.');
    const rating = {task_id: task.task_id, rater_id: rater, rater_kind: 'human',
      review_ref: 'google-form:' + raw.form_id + '/' + raw.response_id,
      evidence: evidence, propositions: task.propositions.map((p, i) => {
        const verdict = Object.prototype.hasOwnProperty.call(OPTIONS.propositions, grid[i]) ? OPTIONS.propositions[grid[i]] : null;
        if (!verdict) throw new Error('알 수 없는 내용 점검 값');
        return {id: p.id, verdict: verdict, evidence: verdict === 'preserved' ? '' : evidence};
      })};
    ['naturalness_vs_input', 'unsupported_addition', 'register_drift'].forEach(field => {
      const value = Object.prototype.hasOwnProperty.call(OPTIONS[field], values[ids[field]]) ? OPTIONS[field][values[ids[field]]] : null;
      if (!value) throw new Error('필수 응답이 없거나 잘못됐습니다: ' + field);
      rating[field] = value;
    });
    const requiresReason = rating.propositions.some(p => p.verdict !== 'preserved')
      || rating.naturalness_vs_input !== 'same' || rating.unsupported_addition !== 'absent'
      || rating.register_drift !== 'absent';
    if (requiresReason && /^(특이사항\s*없음|없음|없다|없어요|none|n\/a|-)\.?$/i.test(evidence))
      throw new Error('변경·문제·평가불가 판단에 구체적인 근거가 필요합니다.');
    return rating;
  });
  return {ratings: ratings, rater: rater, feedback: String(values[mapping.feedback] || '')};
}
function signature_(r) {
  return JSON.stringify([r.propositions.map(p => p.verdict), r.naturalness_vs_input,
    r.unsupported_addition, r.register_drift]);
}
function flags_(r, entry) {
  const unknown = r.propositions.some(p => p.verdict === 'unassessable')
    || [r.naturalness_vs_input, r.unsupported_addition, r.register_drift].includes('unassessable');
  return {unknown: unknown || entry.task.finish_reason !== 'stop',
    error: r.propositions.some(p => ['omitted', 'changed'].includes(p.verdict)) || r.unsupported_addition === 'present'};
}

function aggregate_(studies, tasks, forms, rawRows) {
  const exports = [], issues = [], feedback = [], accepted = new Map();
  const exposures = new Map();
  const taskMap = new Map(tasks.map(r => [r.study_id + '/' + JSON.parse(r.entry_json).task.task_id,
    {entry: JSON.parse(r.entry_json), batch: r.batch_id}]));
  // First complete valid submission per rater/task is counted; all later attempts remain in Raw/Issues.
  rawRows.slice().sort((a, b) => String(a.received_at).localeCompare(String(b.received_at))
    || String(a.response_id).localeCompare(String(b.response_id))).forEach(raw => {
    const f = forms.find(f => f.form_id === raw.form_id);
    if (!f) { issues.push({response_id: raw.response_id, reason: 'unknown form'}); return; }
    const study = studies.find(s => s.study_id === f.study_id);
    if (!study) throw new Error('Unknown study');
    try {
      const normalized = normalize_(raw, f, tasks.filter(t => t.study_id === f.study_id && t.batch_id === f.batch_id)
        .map(t => JSON.parse(t.entry_json)));
      if (normalized.feedback) feedback.push({study_id: f.study_id, response_id: raw.response_id,
        rater_id: normalized.rater, comment: normalized.feedback});
      normalized.ratings.forEach(rating => {
        const id = f.study_id + '/' + rating.task_id + '/' + rating.rater_id;
        if (accepted.has(id)) {
          issues.push({study_id: f.study_id, response_id: raw.response_id, task_id: rating.task_id,
            rater_id: rating.rater_id, reason: signature_(accepted.get(id)) === signature_(rating)
              ? 'duplicate: excluded' : 'conflicting resubmission: excluded; review required'});
          return;
        }
        const entry = taskMap.get(f.study_id + '/' + rating.task_id).entry;
        const exposureKey = f.study_id + '/' + rating.rater_id + '/' + entry.meta.group_id;
        if (exposures.has(exposureKey) && exposures.get(exposureKey) !== f.batch_id)
          issues.push({study_id: f.study_id, response_id: raw.response_id, task_id: rating.task_id,
            rater_id: rating.rater_id, reason: 'related content seen in another batch: interpretation caution'});
        exposures.set(exposureKey, f.batch_id);
        accepted.set(id, rating);
        exports.push({study_id: f.study_id, package_sha256: study.package_sha256, rating_json: JSON.stringify(rating)});
      });
    } catch (error) {
      issues.push({study_id: f.study_id, response_id: raw.response_id, reason: String(error)});
    }
  });
  const groups = new Map(), coverage = [];
  taskMap.forEach(({entry, batch}, taskIdentity) => {
    const studyId = taskIdentity.slice(0, taskIdentity.indexOf('/'));
    const study = studies.find(s => s.study_id === studyId);
    const cellKey = JSON.stringify([studyId, entry.meta.candidate, entry.meta.expected_edit]);
    if (!groups.has(cellKey)) groups.set(cellKey, {study_id: studyId, candidate: entry.meta.candidate,
      edit_expectation: entry.meta.expected_edit, tasks: 0, rated_tasks: 0, two_raters: 0,
      disagreements: 0, ratings: 0, meaning_error_ratings: 0, register_drift_ratings: 0,
      improved: 0, same: 0, worse: 0, unassessable_ratings: 0});
    const cell = groups.get(cellKey);
    const rs = [...accepted.entries()].filter(([k]) => k.startsWith(taskIdentity + '/')).map(([, v]) => v);
    const disagreement = new Set(rs.map(signature_)).size > 1;
    cell.tasks++; cell.rated_tasks += Number(rs.length > 0); cell.two_raters += Number(rs.length >= 2);
    cell.disagreements += Number(disagreement); cell.ratings += rs.length;
    rs.forEach(r => {
      const flags = flags_(r, entry);
      cell.meaning_error_ratings += Number(flags.error);
      cell.register_drift_ratings += Number(r.register_drift === 'present');
      cell.unassessable_ratings += Number(flags.unknown);
      if (['improved', 'same', 'worse'].includes(r.naturalness_vs_input)) cell[r.naturalness_vs_input]++;
    });
    coverage.push({study_id: studyId, batch_id: batch, task_id: entry.task.task_id, raters: rs.length,
      target_raters: study.target_raters, disagreement: disagreement,
      status: disagreement ? '의견 차이 검토' : rs.length < study.target_raters ? '추가 평가 필요' : '평가 수 충족 (승인 아님)'});
  });
  return {Export: exports, Issues: issues, Feedback: feedback, Dashboard: [...groups.values()], Coverage: coverage};
}
function rebuild_(ss) {
  const result = aggregate_(rows_(ss, 'Studies'), rows_(ss, 'Tasks'), rows_(ss, 'Forms'), rows_(ss, 'Raw'));
  Object.keys(result).forEach(name => replace_(ss, name, result[name]));
}
function retryErrors() {
  requireOwner_();
  const lock = LockService.getScriptLock(); lock.waitLock(30000);
  try {
    const ss = workbook_();
    rows_(ss, 'Forms').forEach((f, i) => {
      if (f.status === 'error' && !f.items_json) { f.status = 'pending'; f.error = ''; updateForm_(ss, i, f); }
      else if (['error', 'publish_error', 'ready'].includes(f.status) && f.items_json && !f.responder_url) {
        const form = FormApp.openById(f.form_id);
        form.setPublished(true); form.setAcceptingResponses(true);
        f.status = 'ready'; f.error = ''; f.responder_url = form.getPublishedUrl(); updateForm_(ss, i, f);
      }
    });
  } finally { lock.releaseLock(); }
}
/** Pause automatic collection/creation without deleting any data or closing forms. */
function pauseAutomation() {
  requireOwner_();
  ScriptApp.getProjectTriggers().filter(t => t.getHandlerFunction() === 'tick_')
    .forEach(t => ScriptApp.deleteTrigger(t));
}

function requireOwner_() {
  const active = Session.getActiveUser().getEmail();
  if (!active || active !== Session.getEffectiveUser().getEmail())
    throw new Error('관리 기능은 소유자가 Apps Script 편집기에서 실행해야 합니다.');
}
function escapeHtml_(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
}
/** Optional web-app entry page. Read-only; model labels and ratings never leave the workbook. */
function doGet() {
  const id = PropertiesService.getScriptProperties().getProperty('KH_WORKBOOK_ID');
  if (!id) return HtmlService.createHtmlOutput('담당자가 설문을 준비 중입니다.');
  const ss = SpreadsheetApp.openById(id);
  const studies = rows_(ss, 'Studies').filter(s => s.status === 'active');
  const forms = rows_(ss, 'Forms');
  let body = '<!doctype html><html lang="ko"><head><meta name="viewport" content="width=device-width,initial-scale=1">'
    + '<style>body{font:17px/1.65 system-ui;background:#f3f6fb;color:#172033;max-width:850px;margin:32px auto;padding:20px}'
    + 'a{display:inline-block;background:#fff;color:#234baa;border:1px solid #d4ddea;border-radius:10px;padding:14px 22px;margin:6px;text-decoration:none}'
    + 'section{margin-top:32px}small{color:#506078}</style></head><body><h1>한국어 문장 다듬기 평가</h1>'
    + '<p>담당자가 안내한 평가 묶음을 선택해 주세요. 원문과 수정문을 직접 읽고 자연스러움·내용 보존·어투를 평가합니다.</p>'
    + '<p>같은 평가자 코드를 계속 사용하고, 이미 제출한 묶음은 다시 제출하지 마세요. 수정이 필요하면 담당자에게 알려 주세요.</p>';
  studies.forEach(study => {
    const ready = forms.filter(f => f.study_id === study.study_id && f.status === 'ready' && f.responder_url);
    body += '<section><h2>' + escapeHtml_(study.title) + '</h2><small>실험 ' + escapeHtml_(study.study_id) + '</small><div>';
    if (!ready.length) body += '<p>설문을 준비 중입니다. 잠시 후 다시 방문해 주세요.</p>';
    ready.forEach(f => {
      // Only Google Forms responder URLs from the creation API are allowed.
      if (!/^https:\/\/docs\.google\.com\/forms\//.test(f.responder_url)) return;
      body += '<a target="_blank" rel="noopener noreferrer" href="' + escapeHtml_(f.responder_url) + '">'
        + escapeHtml_(f.batch_id) + '</a>';
    });
    body += '</div></section>';
  });
  body += '<p><small>개인정보는 입력하지 마세요. 응답은 문장 품질 연구·개발에 사용됩니다. 평가 결과와 다른 사람의 응답은 공개하지 않습니다.</small></p></body></html>';
  return HtmlService.createHtmlOutput(body).setTitle('한국어 문장 평가');
}

function guide_(ss) {
  const name = '시작안내';
  const sheet = ss.getSheetByName(name) || ss.insertSheet(name, 0);
  const values = [
    ['한국어 문장 평가 관리', '사용 안내'],
    ['설문 링크', 'Forms 탭의 responder_url만 평가자에게 공유하세요. 1분마다 한 묶음씩 생성됩니다.'],
    ['공유 링크 하나', 'Apps Script에서 웹 앱으로 배포하면 모든 활성 실험의 설문 입구가 만들어집니다.'],
    ['평가자 배정', '사람마다 고유 코드를 주고 묶음마다 서로 다른 두 사람에게 배정하세요. 반복 내용 노출은 Issues에 표시됩니다.'],
    ['Dashboard', '실험·조건별 평가 수와 의견 차이, 의미 오류, 자연스러움 판단 횟수입니다. 평균 품질 점수나 자동 승자가 아닙니다.'],
    ['Coverage', '평가가 부족하거나 의견이 갈린 문항을 확인하세요. 같은 사람의 중복 제출은 인원수에 더하지 않습니다.'],
    ['Issues', '불완전 응답·근거 부족·중복·충돌을 확인하세요. Raw는 보존하며 첫 유효 응답만 집계합니다.'],
    ['Feedback', '설문 자체에 대한 자유 의견입니다.'],
    ['Export', '이 탭을 CSV로 내려받으면 프로젝트의 명제별 평가 집계기에 다시 넣을 수 있습니다.'],
    ['응답 반영', '자동으로 약 5분마다 확인합니다. 즉시 반영하려면 Apps Script에서 refresh를 실행하세요. 규모가 커지면 여러 주기에 나눕니다.'],
    ['실험 종료', 'Apps Script에서 closeCurrentStudy를 실행하면 현재 설치 코드의 실험만 응답을 마감합니다. 기존 결과는 보존됩니다.'],
    ['자동화 중지', 'Apps Script에서 pauseAutomation을 실행하세요. 재개는 setup입니다.'],
    ['공유 범위', '이 관리 시트와 Apps Script에는 모델 대응표가 있습니다. 평가자에게 공유하지 마세요.'],
    ['평가 한계', '평가자 코드는 담당자가 배정합니다. 코드 두 개가 실제 두 사람임을 시스템이 증명하지는 않습니다.'],
    ['수정·철회 요청', '원본 Google Forms 응답과 Raw의 해당 response_id를 담당자가 함께 처리한 뒤 refresh하세요. 파생 집계는 다시 만들어집니다.']
  ];
  sheet.clearContents(); sheet.getRange(1, 1, values.length, 2).setValues(values).setWrap(true);
  sheet.setColumnWidth(1, 180); sheet.setColumnWidth(2, 700);
  sheet.getRange(1, 1, 1, 2).setFontWeight('bold');
}

/** Close this package's study; preserve all forms, responses and other studies. */
function closeCurrentStudy() {
  requireOwner_();
  if (!SURVEY_PACKAGE) throw new Error('Install.gs를 사용하세요.');
  const lock = LockService.getScriptLock(); lock.waitLock(30000);
  try {
    const ss = workbook_();
    const id = SURVEY_PACKAGE.study_id;
    rows_(ss, 'Forms').filter(f => f.study_id === id && f.form_id)
      .forEach(f => FormApp.openById(f.form_id).setAcceptingResponses(false));
    sync_(ss, new Set([id]));
    const studies = rows_(ss, 'Studies');
    studies.forEach(s => { if (s.study_id === id) s.status = 'closed'; });
    replace_(ss, 'Studies', studies); rebuild_(ss);
    console.log('응답 마감: ' + id);
  } finally { lock.releaseLock(); }
}

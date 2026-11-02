// Fake responses only. This does not test Google authorization or native Forms APIs.
const {readFileSync} = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const context = vm.createContext({console});
vm.runInContext(readFileSync('integrations/google_forms/Code.gs', 'utf8'), context);
function fixture() {
  const entry = {task: {task_id:'t1', input:'입력', output:'출력', finish_reason:'stop',
    propositions:[{id:'p1',claim:'확인했다.'}]}, meta:{candidate:'secret-model',expected_edit:'minimal',group_id:'g1'}};
  const study = {study_id:'study-1', package_sha256:'hash',target_raters:2};
  const mapping = {rater:'code',consent:'consent',feedback:'feedback',tasks:[{task_id:'t1',
    naturalness_vs_input:'n',propositions:'p',unsupported_addition:'a',register_drift:'r',evidence:'e'}]};
  const form = {study_id:'study-1', batch_id:'batch-1',form_id:'form1', items_json:JSON.stringify(mapping)};
  const task = {study_id:'study-1',batch_id:'batch-1',entry_json:JSON.stringify(entry)};
  const values = {code:'rater-01',consent:['한국어 글을 직접 읽고 평가하며, 응답의 연구·개발 이용에 동의합니다.'],
    n:'비슷함',p:['유지됨'],a:'없음',r:'없음',e:'특이사항 없음',feedback:'=1+1'};
  const raw = {form_id:'form1',response_id:'response-1',received_at:'2026-10-22T10:00:00Z',response_json:JSON.stringify(values)};
  return {entry,study,form,task,values,raw};
}
{
  const f=fixture();
  const result=context.aggregate_([f.study],[f.task],[f.form],[f.raw]);
  assert.equal(result.Export.length,1); assert.equal(result.Coverage[0].raters,1);
  assert.equal(result.Dashboard[0].meaning_error_ratings,0);
  assert.equal(result.Coverage[0].status,'추가 평가 필요');
  assert.equal(context.safeCell_(result.Feedback[0].comment),"'=1+1");
}
{
  const f=fixture();
  const raw2={...f.raw,response_id:'response-2'};
  const r=context.aggregate_([f.study],[f.task],[f.form],[raw2,f.raw]);
  assert.equal(r.Export.length,1);assert.equal(r.Issues.length,1);assert.equal(r.Coverage[0].raters,1);
  const second={...f.values,code:'rater-02',p:['빠짐'],e:'확인했다는 사실이 출력에 없다.'};
  raw2.response_json=JSON.stringify(second);
  const disagree=context.aggregate_([f.study],[f.task],[f.form],[raw2,f.raw]);
  assert.equal(disagree.Coverage[0].disagreement,true);
  assert.equal(disagree.Dashboard[0].meaning_error_ratings,1);
  assert.equal(disagree.Dashboard[0].two_raters,1);
}
{
  const f=fixture();
  for (const patch of [{p:[]},{n:'constructor'},{consent:[]},{p:['빠짐'],e:'없음'}]) {
    const raw={...f.raw,response_json:JSON.stringify({...f.values,...patch})};
    const result=context.aggregate_([f.study],[f.task],[f.form],[raw]);
    assert.equal(result.Export.length,0);assert.equal(result.Issues.length,1);
    assert.equal(result.Dashboard[0].rated_tasks,0);
  }
  const unknown={...f.values,n:'판단 어려움',e:'문장이 불완전해 판단하기 어렵다.'};
  const result=context.aggregate_([f.study],[f.task],[f.form],[{...f.raw,response_json:JSON.stringify(unknown)}]);
  assert.equal(result.Dashboard[0].unassessable_ratings,1);
  assert.equal(result.Dashboard[0].same,0);
}
{
  const f=fixture();
  const other={...f.study,study_id:'study-2'};
  const task2={...f.task,study_id:'study-2'};
  const result=context.aggregate_([f.study,other],[f.task,task2],[f.form],[f.raw]);
  assert.equal(result.Dashboard.length,2);
  assert.equal(result.Dashboard.find(x=>x.study_id==='study-2').rated_tasks,0);
}
// Public functions must not expose admin operations to an anonymous web-app visitor.
{
  context.Session={getActiveUser:()=>({getEmail:()=>''}),getEffectiveUser:()=>({getEmail:()=> 'owner@example.test'})};
  for (const name of ['setup','refresh','retryErrors','pauseAutomation','closeCurrentStudy']) assert.throws(()=>context[name](),/소유자/);
  assert.equal(context.escapeHtml_('<script>"&'), '&lt;script&gt;&quot;&amp;');
}
console.log('Google Forms pure-logic checks passed (mock responses; no Google calls).');

// The public landing page contains only responder links, never editor URLs, models or ratings.
{
  context.PropertiesService={getScriptProperties:()=>({getProperty:()=> 'private-workbook'})};
  context.SpreadsheetApp={openById:()=>({getSheetByName:(name)=>({getDataRange:()=>({getValues:()=>{
    if(name==='Studies') return [['study_id','package_sha256','title','tasks','target_raters','status'],
      ['study-1','secret-hash','평가',1,2,'active']];
    if(name==='Forms') return [['study_id','batch_id','form_id','status','responder_url','edit_url','items_json','error'],
      ['study-1','batch-001','id','ready','https://docs.google.com/forms/d/e/public/viewform','https://private-editor','secret-model','']];
    throw new Error('public page must not read '+name);
  }})})})};
  context.HtmlService={createHtmlOutput:(body)=>({body,setTitle(){return this;}})};
  const body=context.doGet().body;
  assert.ok(body.includes('/public/viewform'));
  for(const secret of ['secret-model','secret-hash','private-editor','private-workbook']) assert.ok(!body.includes(secret));
}

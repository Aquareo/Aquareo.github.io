import { z } from 'zod';

const short = z.string().max(300);
const date = z.string().refine(v => v === '' || /^\d{4}-\d{2}-\d{2}$/.test(v) && !Number.isNaN(Date.parse(v)), '日期格式有误');
const link = z.string().max(1500).refine(v => v === '' || /^https:\/\/mail\.google\.com\//.test(v), '邮件链接须为 Gmail HTTPS 链接');
export const appSchema = z.object({
 id: z.string().min(1).max(100), school: short.min(1), person: short.min(1), topic: short,
 status: z.enum(['draft','waiting','positive','meeting','applied','offer','closed']),
 category: z.enum(['application','advice']), latest: date, deadline: date,
 next: z.string().max(2000), notes: z.string().max(6000), source: link,
});
export const eventSchema = z.object({
 id: z.string().min(1).max(100), appId: z.string().max(100), date: date.refine(Boolean),
 title: short.min(1), detail: z.string().max(6000), type: z.enum(['sent','reply','meeting','note','draft','application','result']),
 source: link, origin: z.enum(['email','manual','sync']),
});
export const taskSchema = z.object({
 id: z.string().min(1).max(100), appId: z.string().max(100), title: short.min(1), due: date,
 detail: z.string().max(2000), done: z.boolean(), source: link,
});
export const stateSchema = z.object({
 schemaVersion: z.literal(1), importedAt: short, importNote: z.string().max(2000),
 sync: z.object({checkedAt:z.string(),status:z.enum(['ok','error']),newMessages:z.number().int().min(0),summary:z.array(z.string().max(2000)).max(20)}).optional(),
 applications: z.array(appSchema).max(300), events: z.array(eventSchema).max(3000), tasks: z.array(taskSchema).max(1000),
}).superRefine((v,ctx)=>{
 for(const key of ['applications','events','tasks'] as const) {
  const ids=v[key].map(x=>x.id);
  if(new Set(ids).size!==ids.length)ctx.addIssue({code:'custom',message:'记录 ID 重复',path:[key]});
 }
 const ids=new Set(v.applications.map(x=>x.id));
 for(const key of ['events','tasks'] as const)for(const item of v[key])if(item.appId&&!ids.has(item.appId))ctx.addIssue({code:'custom',message:'关联申请不存在',path:[key]});
});
export type TrackerState = z.infer<typeof stateSchema>;
export type Application = z.infer<typeof appSchema>;
export type TimelineEvent = z.infer<typeof eventSchema>;
export type Task = z.infer<typeof taskSchema>;
export const statusNames: Record<Application['status'], string> = {draft:'草稿 · 未发送',waiting:'已联系 · 待回复',positive:'鼓励申请',meeting:'交流安排中',applied:'已提交申请',offer:'已获录取',closed:'已结束'};
export const eventNames: Record<TimelineEvent['type'], string> = {sent:'发出联系',reply:'收到回复',meeting:'交流安排',note:'个人记录',draft:'草稿',application:'提交申请',result:'结果'};

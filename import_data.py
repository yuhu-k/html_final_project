import json
import os
from datetime import datetime,timedelta


jsonfile = open('html.2023.final.data/demographic.json','r',encoding="utf-8")
sta_info = json.load(jsonfile)
jsonfile.close()

data_file_name = 'data2.csv'

if os.path.isfile(data_file_name):
  os.remove(data_file_name)
f = open(data_file_name,'w')
f.write('oneWeekAgoBikeNumbefore2,oneWeekAgoBikeNumbefore1,oneWeekAgoBikeNum,oneWeekAgoBikeNumafter1,oneWeekAgoBikeNumafter2,month,day,weekday,hour,min,latitude,longitude,total,standby\n')

parent_dir_name = 'html.2023.final.data/release'
i = 0
for date in os.listdir(parent_dir_name):
  print('%d/%d\r' % (i,os.listdir(parent_dir_name).__len__()))
  i += 1
  dt_1 = datetime.strptime(date, '%Y%m%d')
  #if dt_1 > datetime.strptime('20231115','%Y%m%d') or dt_1 < datetime.strptime('20231102','%Y%m%d'):
  #   continue
  for sta_id in os.listdir('%s/%s' % (parent_dir_name,date)):
    input_file = open('%s/%s/%s' % (parent_dir_name,date,sta_id),'r',encoding='utf-8')
    sta_usage = json.load(input_file)
    input_file.close()

    if   os.path.exists('%s/%s/%s' % (parent_dir_name,(dt_1-timedelta(days=7)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s' % (parent_dir_name,(dt_1-timedelta(days=7)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s' % (parent_dir_name,(dt_1+timedelta(days=7)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s' % (parent_dir_name,(dt_1+timedelta(days=7)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s' % (parent_dir_name,(dt_1-timedelta(days=14)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s' % (parent_dir_name,(dt_1-timedelta(days=14)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s' % (parent_dir_name,(dt_1+timedelta(days=14)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s' % (parent_dir_name,(dt_1+timedelta(days=14)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    else:
      input_file = open('%s/%s/%s' % (parent_dir_name,date,sta_id),'r',encoding='utf-8')
    sta_usage2 = json.load(input_file)
    input_file.close()
    

    sta_id = sta_id.split('.')[0]

    for time in sta_usage:
      if sta_usage[time] != {}:
        dt = datetime.strptime(date+' '+time,'%Y%m%d %H:%M')
        bikeNum = ''
        if (dt.strftime('%H') == '00' and int(dt.strftime('%M')) <= 10):
          for k in range(5):
            if sta_usage2[(dt + timedelta(minutes=k)).strftime('%H:%M')] != {}:
              time2 = (dt + timedelta(minutes=k)).strftime('%H:%M')
              bikeNum = '%s,%s,%s,%s,%s,' % (sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'])
        elif (dt.strftime('%H') == '23' and int(dt.strftime('%M')) >= 50):
          for k in range(5):
            if sta_usage2[(dt - timedelta(minutes=k)).strftime('%H:%M')] != {}:
              time2 = (dt - timedelta(minutes=k)).strftime('%H:%M')
              bikeNum = '%s,%s,%s,%s,%s,' % (sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'])
        else:
          for k in range(5):
            d = dt + timedelta(minutes=k-2)
            for j in range(5):
              if sta_usage2[(d+timedelta(minutes=j)).strftime('%H:%M')] != {}:
                bikeNum += str(sta_usage2[(d+timedelta(minutes=j)).strftime('%H:%M')]['sbi'])  + ','
                break
            if j == 5:
              print('error\n\n')
              exit()


        f.write('%s%s,%s,%s,%s,%s\n' % (bikeNum,dt.strftime("%m,%d,%w,%H,%M"),sta_info[sta_id]['lat'],sta_info[sta_id]['lng'],sta_usage[time]['tot'],sta_usage[time]['sbi']))
        

f.close()

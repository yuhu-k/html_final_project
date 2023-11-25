import os
from datetime import datetime,timedelta
import json
stage1 = open('sample_submission_stage1.csv','r',encoding='utf-8')
output = open('stage1_input.csv','w',encoding='utf-8')
output.write('oneWeekAgoBikeNumbefore2,oneWeekAgoBikeNumbefore1,oneWeekAgoBikeNum,oneWeekAgoBikeNumafter1,oneWeekAgoBikeNumafter2,month,day,weekday,hour,min,latitude,longitude,total\n')



jsonfile = open('html.2023.final.data/demographic.json','r',encoding="utf-8")
sta_info = json.load(jsonfile)
jsonfile.close()

parent_dir_name = 'html.2023.final.data/release'
content = stage1.readlines()[1:]
i = 0
for line in content:
    print('%d/%d\r' % (i,content.__len__()))
    i += 1
    line = line.split(',')
    id = line[0]
    detail = id.split('_')
    date = detail[0]
    sta_id = detail[1]
    time = detail[2]

    dt_1 = datetime.strptime(date, '%Y%m%d')

    if   os.path.exists('%s/%s/%s.json' % (parent_dir_name,(dt_1-timedelta(days=7)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s.json' % (parent_dir_name,(dt_1-timedelta(days=7)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s.json' % (parent_dir_name,(dt_1+timedelta(days=7)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s.json' % (parent_dir_name,(dt_1+timedelta(days=7)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s.json' % (parent_dir_name,(dt_1-timedelta(days=14)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s.json' % (parent_dir_name,(dt_1-timedelta(days=14)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s.json' % (parent_dir_name,(dt_1+timedelta(days=14)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s.json' % (parent_dir_name,(dt_1+timedelta(days=14)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s.json' % (parent_dir_name,(dt_1-timedelta(days=21)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s.json' % (parent_dir_name,(dt_1-timedelta(days=21)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s.json' % (parent_dir_name,(dt_1+timedelta(days=21)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s.json' % (parent_dir_name,(dt_1+timedelta(days=21)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s.json' % (parent_dir_name,(dt_1-timedelta(days=28)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s.json' % (parent_dir_name,(dt_1-timedelta(days=28)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    elif os.path.exists('%s/%s/%s.json' % (parent_dir_name,(dt_1+timedelta(days=28)).strftime('%Y%m%d'),sta_id)):
      input_file = open('%s/%s/%s.json' % (parent_dir_name,(dt_1+timedelta(days=28)).strftime('%Y%m%d'),sta_id),'r',encoding='utf-8')
    else:
      print('error1\n\n')
      exit()
    sta_usage2 = json.load(input_file)
    input_file.close()

    dt = datetime.strptime(date+' '+time,'%Y%m%d %H:%M')
    bikeNum = ''
    tot = 0
    if (dt.strftime('%H') == '00' and int(dt.strftime('%M')) <= 10):
      for k in range(5):
        if sta_usage2[(dt + timedelta(minutes=k)).strftime('%H:%M')] != {}:
          time2 = (dt + timedelta(minutes=k)).strftime('%H:%M')
          bikeNum = '%s,%s,%s,%s,%s,' % (sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'])
          tot = sta_usage2[time2]['tot']
    elif (dt.strftime('%H') == '23' and int(dt.strftime('%M')) >= 50):
      for k in range(5):
        if sta_usage2[(dt - timedelta(minutes=k)).strftime('%H:%M')] != {}:
          time2 = (dt - timedelta(minutes=k)).strftime('%H:%M')
          bikeNum = '%s,%s,%s,%s,%s,' % (sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'],sta_usage2[time2]['sbi'])
          tot = sta_usage2[time2]['tot']
    else:
      for k in range(5):
        d = dt + timedelta(minutes=k-2)
        for j in range(5):
          if sta_usage2[(d+timedelta(minutes=j)).strftime('%H:%M')] != {}:
            t = (d+timedelta(minutes=j)).strftime('%H:%M')
            bikeNum += str(sta_usage2[t]['sbi'])  + ','
            tot = sta_usage2[t]['tot']
            break
        if j == 5:
          print('error2\n\n')
          exit()
    
    output.write('%s%s,%s,%s,%s\n' % (bikeNum,dt.strftime("%m,%d,%w,%H,%M"),sta_info[sta_id]['lat'],sta_info[sta_id]['lng'],tot))
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import yaml
from server import config_server as cs
from server import dashboard_server as ds


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.stack = __import__('contextlib').ExitStack()
        for module in [cs, ds]:
            for field, path in [('ROOT', self.root), ('CONFIG_PATH', self.root/'config.yaml'), ('SAVED_CONFIGS_PATH', self.root/'saved.json')]:
                self.stack.enter_context(patch.object(module, field, path))
        self.stack.enter_context(patch.object(cs, 'TAX_PATH', self.root/'taxonomy.yaml'))
        self.stack.enter_context(patch.object(ds, 'TAXONOMY_PATH', self.root/'taxonomy.yaml'))
        self.client = cs.app.test_client()
        self.dashboard = ds.app.test_client()
        self.cfg = {'project': {'name':'A', 'location':'City', 'year_range':[2020,2030]}, 'paths':{'input_dir':str(self.root),'output_csv':'outputs/a.csv'}, 'processing': {'provider':'vectorengine','model':'test'}}
        self.tax = {'domains':[{'name':'A domain'}]}

    def tearDown(self):
        self.stack.close()
        self.tmp.cleanup()

    def save(self, name='A', original=None):
        cfg = copy.deepcopy(self.cfg)
        cfg['project']['name'] = name
        cfg['paths']['output_csv'] = f'outputs/{name}.csv'
        return self.client.post('/save-project', json={'config':cfg,'taxonomy':{'domains':[{'name':name+' domain'}]},'original_name':original})

    def test_save_rename_and_activate_restore_taxonomy(self):
        a = self.save().json
        self.assertTrue(a['ok'])
        original_id = a['config']['project']['id']
        self.assertEqual(self.save().status_code, 409)
        self.save('B')
        activated = self.client.post('/activate-project',json={'name':'A'}).json
        self.assertEqual(activated['config']['project']['name'], 'A')
        self.assertEqual(yaml.safe_load((self.root/'taxonomy.yaml').read_text()), self.tax)
        renamed = self.save('C', 'A').json
        self.assertEqual(renamed['config']['project']['id'], original_id)
        self.assertNotIn('A', json.loads((self.root/'saved.json').read_text()))

    def test_failed_save_does_not_change_active_project(self):
        self.save()
        before=(self.root/'config.yaml').read_bytes()
        cfg=copy.deepcopy(self.cfg);cfg['project']['name']='B';cfg['paths']['input_dir']='/missing/folder'
        r=self.client.post('/save-project',json={'config':cfg,'taxonomy':self.tax})
        self.assertEqual(r.status_code,400)
        self.assertEqual((self.root/'config.yaml').read_bytes(),before)

    def test_output_collision_rejected(self):
        self.save()
        cfg=copy.deepcopy(self.cfg);cfg['project']['name']='B';cfg['paths']['output_csv']='outputs/A.csv'
        self.assertEqual(self.client.post('/save-project',json={'config':cfg,'taxonomy':self.tax}).status_code,400)

    def test_review_isolation_and_export(self):
        paths=[]
        for name in ['a','b']:
            path=self.root/(name+'.csv'); paths.append(path)
            pd.DataFrame([{'filename':'same.txt','file_path':'/one/same.txt','domain':'Original'}, {'filename':'same.txt','file_path':'/two/same.txt','domain':'Original'}]).to_csv(path,index=False)
        result=self.dashboard.post('/api/review',json={'csv':str(paths[0]),'file_path':'/one/same.txt','edits':{'domain':'Edited'}})
        self.assertTrue(result.json['ok'])
        self.assertEqual(self.dashboard.get('/api/reviewed',query_string={'csv':str(paths[1])}).json['reviewed'],[])
        rows=self.dashboard.get('/api/data',query_string={'csv':str(paths[0])}).json['rows']
        self.assertEqual([r['domain'] for r in rows],['Edited','Original'])
        exported=self.dashboard.get('/api/download-csv',query_string={'csv':str(paths[0])}).text
        self.assertIn('Edited',exported)
        self.assertNotIn('Edited',paths[0].read_text())

    def test_navigation_uses_this_project_ports(self):
        self.cfg['configuration'] = {'port': 5189}
        self.cfg['dashboard'] = {'port': 5190}
        (self.root/'config.yaml').write_text(yaml.safe_dump(self.cfg))
        self.assertEqual(self.dashboard.get('/project-ui?page=settings').headers['Location'], 'http://localhost:5189/wizard.html')
        self.assertEqual(self.client.get('/results-ui').headers['Location'], 'http://localhost:5190/')
        self.assertEqual(self.dashboard.get('/project-ui?page=https://invalid.example').headers['Location'], 'http://localhost:5189/')

    def test_historical_ai_requires_own_snapshot(self):
        path=self.root/'old.csv'
        with self.assertRaises(ValueError): ds._result_config(path)
        Path(str(path)+'.config.yaml').write_text(yaml.safe_dump(self.cfg))
        self.assertEqual(ds._result_config(path)['project']['name'],'A')


if __name__=='__main__': unittest.main()

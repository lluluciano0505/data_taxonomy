import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from core.api_connection import connection_config, get_api_key, normalize_base_url
from core.config_loader import get_paths_config, get_processing_config
from server import config_server as server


class ConnectionTests(unittest.TestCase):
    def test_legacy_provider_inference(self):
        for host, provider in [('api.deepseek.com', 'deepseek'), ('api.vectorengine.ai', 'vectorengine'), ('api.vectorengine.cn', 'vectorengine'), ('openrouter.ai', 'openrouter')]:
            self.assertEqual(get_processing_config({'processing': {'base_url': f'https://{host}/v1'}}).provider, provider)

    def test_url_normalization(self):
        for suffix in ['', '/', '/v1/', '/v1/chat/completions']:
            self.assertEqual(normalize_base_url('https://api.vectorengine.ai' + suffix), 'https://api.vectorengine.ai/v1')
        for url in ['file:///tmp/a', 'https://user:pass@example.org', 'https://example.org?key=secret']:
            with self.assertRaises(ValueError):
                normalize_base_url(url)

    def test_keys_are_isolated(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'deep', 'OPENROUTER_API_KEY_BACKUP': 'backup'}, clear=True):
            self.assertEqual(get_api_key('vectorengine')[0], '')
            self.assertEqual(get_api_key('deepseek')[0], 'deep')
            self.assertEqual(get_api_key('openrouter')[0], 'backup')

    def test_relative_paths_resolve_from_application_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The loader's application root is the repository root; relative
            # output/taxonomy paths must not depend on the caller's cwd.
            cfg = get_paths_config({'paths': {
                'input_dir': 'project files',
                'output_csv': 'outputs/result.csv',
                'taxonomy_path': 'taxonomies/project.yaml',
            }})
            app_root = Path(__file__).resolve().parents[1]
            self.assertEqual(cfg.input_dir, (app_root / 'project files').absolute())
            self.assertEqual(cfg.output_csv, (app_root / 'outputs/result.csv').absolute())
            self.assertEqual(cfg.taxonomy_path, str((app_root / 'taxonomies/project.yaml').absolute()))

    def test_connection_endpoints(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            with patch.object(server, 'ROOT', root), patch.object(server, 'CONFIG_PATH', root / 'config.yaml'):
                client = server.app.test_client()
                cfg = {'processing': {'provider': 'vectorengine', 'base_url': 'https://api.vectorengine.ai/v1/chat/completions', 'model': 'test-model'}}
                self.assertEqual(client.post('/save-config', json=cfg).status_code, 200)
                self.assertEqual(client.get('/load-config').json['processing']['base_url'], 'https://api.vectorengine.ai/v1')
                self.assertEqual(client.post('/test-connection', json=cfg).status_code, 400)
                self.assertEqual(client.post('/save-apikey', json={'provider': 'vectorengine', 'key': 'test-secret'}).status_code, 200)
                self.assertFalse(client.get('/apikey-status?provider=openrouter').json['set'])
                self.assertNotIn('test-secret', client.get('/apikey-status?provider=vectorengine').text)
                self.assertNotIn('test-secret', (root / 'config.yaml').read_text())
                response = Mock(status_code=200)
                response.json.return_value = {'choices': [{'message': {'content': 'OK'}}]}
                with patch.object(server.http_requests, 'post', return_value=response) as post:
                    self.assertTrue(client.post('/test-connection', json=cfg).json['ok'])
                    self.assertEqual(post.call_args.args[0], 'https://api.vectorengine.ai/v1/chat/completions')
                    self.assertEqual(post.call_args.kwargs['headers']['Authorization'], 'Bearer test-secret')
                    response.status_code = 401
                    self.assertEqual(client.post('/test-connection', json=cfg).status_code, 400)
                self.assertEqual(client.post('/save-apikey', json={'provider': 'bad', 'key': 'x'}).status_code, 400)
                self.assertEqual(client.post('/save-apikey', json={'provider': 'vectorengine', 'key': 'x\nBAD=y'}).status_code, 400)


if __name__ == '__main__':
    unittest.main()

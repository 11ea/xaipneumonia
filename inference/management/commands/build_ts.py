# inference/management/commands/build_ts.py
from django.core.management.base import BaseCommand
import subprocess
from pathlib import Path

class Command(BaseCommand):
    help = 'Build TypeScript files for Django'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--watch',
            action='store_true',
            help='Watch for changes and rebuild automatically'
        )
    
    def handle(self, *args, **options):
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        
        try:
            if options['watch']:
                self.stdout.write('Starting TypeScript watch mode...')
                subprocess.run(['npm', 'run', 'dev'], cwd=project_root)
            else:
                self.stdout.write('Building TypeScript files...')
                result = subprocess.run(
                    ['npm', 'run', 'build'],
                    capture_output=True,
                    text=True,
                    cwd=project_root
                )
                
                if result.returncode == 0:
                    self.stdout.write(
                        self.style.SUCCESS('TypeScript build completed successfully!')
                    )
                else:
                    self.stderr.write(
                        self.style.ERROR(f'Build failed: {result.stderr}')
                    )
                    
        except FileNotFoundError:
            self.stderr.write(
                self.style.ERROR('npm not found. Please install Node.js and npm.')
            )
        except KeyboardInterrupt:
            self.stdout.write('\nBuild process interrupted.')
import os
import requests
import git
from git import GitCommandError

REPO_OWNER = 'sefaakkoc'
REPO_NAME = 'molytica'
current_directory = os.getcwd()
REPO_DIR = os.path.join(current_directory)
API_URL = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest'

def check_for_update():
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            latest_release = response.json()
            return latest_release['tag_name']
        else:
            print(f"GitHub API error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API connection error: {e}")
        return None

def check_local_changes():
    try:
        repo = git.Repo(REPO_DIR)
        return repo.is_dirty() or len(repo.untracked_files) > 0
    except git.InvalidGitRepositoryError:
        print("This directory is not a Git repository!")
        return False

def get_current_version():
    try:
        repo = git.Repo(REPO_DIR)
        return repo.head.commit.hexsha[:7]
    except Exception as e:
        print(f"Error getting current version: {e}")
        return None

def update_repository():
    if not os.path.exists(REPO_DIR):
        print(f"Repository directory not found: {REPO_DIR}")
        return False
    try:
        repo = git.Repo(REPO_DIR)

        if check_local_changes():
            print("There are uncommitted changes! Please commit or stash them first.")
            return False
        
        print("Checking for updates...")
        current_commit = get_current_version()
        origin = repo.remotes.origin
        origin.fetch()
        
        if repo.active_branch.name != 'main':
            try:
                repo.git.checkout('main')
            except GitCommandError:
                try:
                    repo.git.checkout('master')
                except GitCommandError:
                    print("Main branch not found (main/master)")
                    return False
        
        try:
            repo.git.merge('origin/main', '--ff-only')
        except GitCommandError:
            try:
                repo.git.merge('origin/master', '--ff-only')
            except GitCommandError:
                print("Merge conflicts detected! Manual intervention required.")
                return False
        
        new_commit = get_current_version()
        if current_commit != new_commit:
            print(f"Update successful! {current_commit} -> {new_commit}")
            return True
        else:
            print("Repository is already up to date!")
            return True
            
    except git.InvalidGitRepositoryError:
        print("This directory is not a Git repository!")
        return False
    except Exception as e:
        print(f"Update error: {e}")
        return False

def run_after_update():
    print("\nRunning main code...")
    try:
        print("Your main application would run here...")
        os.system("echo porno")
        print("Main code executed successfully!")
    except Exception as e:
        print(f"Error running main code: {e}")

def main():
    print("=" * 50)
    print("GitHub Repository Auto-Update")
    print("=" * 50)
    update_success = update_repository()  
    if update_success:
        run_after_update()
    else:
        print("Update failed, main code not executed.")
        print("Please fix the issues and try again.")

if __name__ == "__main__":
    main()
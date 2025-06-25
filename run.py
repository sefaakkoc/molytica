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
            print(f"GitHub API hatası: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API bağlantı hatası: {e}")
        return None

def check_local_changes():
    try:
        repo = git.Repo(REPO_DIR)
        return repo.is_dirty() or len(repo.untracked_files) > 0
    except git.InvalidGitRepositoryError:
        print("Bu dizin bir Git repository değil!")
        return False

def get_current_version():
    try:
        repo = git.Repo(REPO_DIR)
        return repo.head.commit.hexsha[:7]
    except Exception as e:
        print(f"Mevcut versiyon alınurken hata: {e}")
        return None

def update_repository():
    if not os.path.exists(REPO_DIR):
        print(f"Repository dizini bulunamadı: {REPO_DIR}")
        return False
    try:
        repo = git.Repo(REPO_DIR)

        if check_local_changes():
            print("Kaydedilmemiş değişiklikler var! Önce commit yapın veya stash'leyin.")
            return False
        print("Güncellemeler kontrol ediliyor...")
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
                    print("Ana branch bulunamadı (main/master)")
                    return False
        try:
            repo.git.merge('origin/main', '--ff-only')
        except GitCommandError:
            try:
                repo.git.merge('origin/master', '--ff-only')
            except GitCommandError:
                print("Merge konfliktleri var! Manuel müdahale gerekli.")
                return False
        new_commit = get_current_version()
        if current_commit != new_commit:
            print(f"Güncelleme başarılı! {current_commit} -> {new_commit}")
            return True
        else:
            print("Repository zaten güncel!")
            return True
            
    except git.InvalidGitRepositoryError:
        print("Bu dizin bir Git repository değil!")
        return False
    except Exception as e:
        print(f"Güncelleme hatası: {e}")
        return False

def run_after_update():
    print("\nAna kod çalıştırılıyor...")
    try:
        print("Ana uygulamanız burada çalışacak...")
        os.system("python main.py")
        print("Ana kod başarıyla çalıştırıldı!")
    except Exception as e:
        print(f"Ana kod çalıştırma hatası: {e}")

def main():
    print("=" * 50)
    print("GitHub Repository Auto-Update")
    print("=" * 50)
    update_success = update_repository()  
    if update_success:
        run_after_update()
    else:
        print("Güncelleme başarısız, ana kod çalıştırılmadı.")
        print("Lütfen sorunları düzeltin ve tekrar deneyin.")

if __name__ == "__main__":
    main()

import os
import json
from PIL import Image

class PDFCliEditor:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.current_page = 1
        self.text_data = self._load_text_data()
        self.total_pages = len([f for f in os.listdir(output_dir) if f.endswith('.png')])

    def _load_text_data(self):
        json_file = os.path.join(self.output_dir, 'text_data.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_text_data(self):
        json_file = os.path.join(self.output_dir, 'text_data.json')
        with open(json_file, 'w') as f:
            json.dump(self.text_data, f, indent=2)

    def display_page_info(self):
        print(f"\nPage {self.current_page}/{self.total_pages}")
        page_key = f"page_{self.current_page:03d}"
        if page_key in self.text_data:
            print("\nText Boxes:")
            for idx, text_box in enumerate(self.text_data[page_key], 1):
                print(f"{idx}. Text: {text_box['text'][:50]}...")
                print(f"   Coordinates: {text_box['bbox']}")
        else:
            print("No text boxes on this page.")

    def delete_text_box(self, box_number):
        page_key = f"page_{self.current_page:03d}"
        if page_key in self.text_data:
            if 1 <= box_number <= len(self.text_data[page_key]):
                del self.text_data[page_key][box_number - 1]
                self._save_text_data()
                print(f"Text box {box_number} deleted.")
            else:
                print("Invalid text box number.")
        else:
            print("No text boxes on this page.")

    def run(self):
        while True:
            self.display_page_info()
            print("\nCommands:")
            print("n - Next page")
            print("p - Previous page")
            print("d <number> - Delete text box")
            print("q - Quit")

            command = input("\nEnter command: ").strip().lower()

            if command == 'q':
                break
            elif command == 'n':
                if self.current_page < self.total_pages:
                    self.current_page += 1
                else:
                    print("Already at last page.")
            elif command == 'p':
                if self.current_page > 1:
                    self.current_page -= 1
                else:
                    print("Already at first page.")
            elif command.startswith('d '):
                try:
                    box_num = int(command.split()[1])
                    self.delete_text_box(box_num)
                except (IndexError, ValueError):
                    print("Invalid delete command. Use 'd <number>'")

def main(output_dir):
    editor = PDFCliEditor(output_dir)
    editor.run()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide output directory path.")

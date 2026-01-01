import unittest
from agent.tools.definitions import ShellTool, FileTool

class TestTools(unittest.TestCase):
    def test_shell_tool_safety(self):
        tool = ShellTool()
        
        # Test dangerous command
        result = tool.execute(command="rm -rf /")
        self.assertFalse(result.success)
        self.assertIn("blocked", result.error)
        
        # Test valid command
        result = tool.execute(command="echo 'test'")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "test")

    def test_file_tool(self):
        # We should mock OS operations, but for this simple check we can try basic logic
        # or mock os/builtins.open. Let's mock for safety.
        from unittest.mock import patch, mock_open
        
        tool = FileTool()
        
        with patch("builtins.open", mock_open(read_data="content")) as mock_file:
            with patch("os.path.exists", return_value=True):
                result = tool.execute(operation="read", path="/tmp/test.txt")
                self.assertTrue(result.success)
                self.assertEqual(result.output, "content")

if __name__ == '__main__':
    unittest.main()

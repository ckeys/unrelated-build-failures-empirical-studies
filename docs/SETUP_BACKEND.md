# Survey Backend Setup Guide (Google Sheets, ~5 minutes)

This survey uses a free Google Apps Script as backend to save responses to a Google Spreadsheet.

## Step 1: Create a Google Spreadsheet

1. Go to [Google Sheets](https://sheets.google.com) and create a new spreadsheet.
2. Name it `CI Survey Responses`.
3. In Row 1, add these column headers (A1 through J1):

```
timestamp | role | experience | project | q1 | q2 | q3 | q3_other_text | q4 | q5
```

## Step 2: Create the Apps Script

1. In the spreadsheet, go to **Extensions → Apps Script**.
2. Delete any existing code and paste the following:

```javascript
function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var data = JSON.parse(e.postData.contents);
  
  sheet.appendRow([
    data.timestamp || new Date().toISOString(),
    data.role || '',
    data.experience || '',
    data.project || '',
    data.q1 || '',
    data.q2 || '',
    data.q3 || '',
    data.q3_other_text || '',
    data.q4 || '',
    data.q5 || ''
  ]);
  
  return ContentService
    .createTextOutput(JSON.stringify({ status: 'success' }))
    .setMimeType(ContentService.MimeType.JSON);
}
```

3. Click **Save** (Ctrl+S).

## Step 3: Deploy as Web App

1. Click **Deploy → New deployment**.
2. Click the gear icon ⚙️ next to "Select type" → choose **Web app**.
3. Set:
   - **Description**: `CI Survey Backend`
   - **Execute as**: `Me`
   - **Who has access**: `Anyone`
4. Click **Deploy**.
5. Click **Authorize access** → select your Google account → click "Allow".
6. **Copy the Web app URL** (it looks like `https://script.google.com/macros/s/AKfyc.../exec`).

## Step 4: Update the Survey Page

1. Open `docs/index.html`.
2. Find the line:
   ```javascript
   const GOOGLE_APPS_SCRIPT_URL = 'YOUR_GOOGLE_APPS_SCRIPT_URL_HERE';
   ```
3. Replace `YOUR_GOOGLE_APPS_SCRIPT_URL_HERE` with your Web app URL from Step 3.

## Step 5: Push to GitHub and Enable GitHub Pages

```bash
git checkout main
git add docs/
git commit -m "Add developer survey for rebuttal study"
git push origin main
```

Then in your GitHub repo:
1. Go to **Settings → Pages**
2. Under "Source", select **Deploy from a branch**
3. Branch: `main`, Folder: `/docs`
4. Click **Save**

Your survey will be live at:
```
https://ckeys.github.io/unrelated-build-failures-empirical-studies/
```

## Testing

1. Open the survey URL in your browser.
2. Fill in and submit a test response.
3. Check your Google Spreadsheet — the response should appear as a new row.
4. Delete the test row before distributing.

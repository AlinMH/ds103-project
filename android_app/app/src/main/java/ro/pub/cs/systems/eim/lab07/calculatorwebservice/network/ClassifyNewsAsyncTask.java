package ro.pub.cs.systems.eim.lab07.calculatorwebservice.network;

import android.os.AsyncTask;
import android.util.Log;
import android.widget.TextView;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.UnsupportedEncodingException;

import cz.msebera.android.httpclient.client.ClientProtocolException;
import cz.msebera.android.httpclient.client.HttpClient;
import cz.msebera.android.httpclient.client.ResponseHandler;
import cz.msebera.android.httpclient.client.methods.HttpGet;
import cz.msebera.android.httpclient.impl.client.BasicResponseHandler;
import cz.msebera.android.httpclient.impl.client.DefaultHttpClient;
import ro.pub.cs.systems.eim.lab07.calculatorwebservice.general.Constants;

public class ClassifyNewsAsyncTask extends AsyncTask<String, Void, String> {

    private TextView resultTextView;
    private TextView resultClass;
    private TextView resultConfidence;

    public ClassifyNewsAsyncTask(TextView resultTextView, TextView resultClass, TextView resultConfidence) {
        this.resultTextView = resultTextView;
        this.resultClass = resultClass;
        this.resultConfidence = resultConfidence;
    }

    @Override
    protected String doInBackground(String... params) {
        String newsUrl = params[0];

        HttpClient httpClient = new DefaultHttpClient();
        HttpGet httpGet = new HttpGet(Constants.GET_WEB_SERVICE_ADDRESS
                + "?" + Constants.NEWS_URL_ATTRIBUTE + "=" + newsUrl);
        ResponseHandler<String> responseHandlerGet = new BasicResponseHandler();
        try {
            return httpClient.execute(httpGet, responseHandlerGet);
        } catch (ClientProtocolException clientProtocolException) {
            Log.e(Constants.TAG, clientProtocolException.getMessage());
            if (Constants.DEBUG) {
                clientProtocolException.printStackTrace();
            }
        } catch (IOException ioException) {
            Log.e(Constants.TAG, ioException.getMessage());
            if (Constants.DEBUG) {
                ioException.printStackTrace();
            }
        }
        return null;
    }

    @Override
    protected void onPostExecute(String result) {
        try {
            JSONObject jsonObject = new JSONObject(result);
            byte[] titleBytes = jsonObject.getString("title").getBytes("ISO-8859-1");
            String title = new String(titleBytes, "UTF-8");
            double confidence = jsonObject.getDouble("confidence");
            String newsClass = jsonObject.getString("class_");

            resultTextView.setText(title);
            resultConfidence.setText(Double.toString(confidence));
            resultClass.setText(newsClass);

        } catch (JSONException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }

    }

}

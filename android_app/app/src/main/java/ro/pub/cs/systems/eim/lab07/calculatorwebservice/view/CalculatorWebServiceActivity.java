package ro.pub.cs.systems.eim.lab07.calculatorwebservice.view;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import ro.pub.cs.systems.eim.lab07.calculatorwebservice.R;
import ro.pub.cs.systems.eim.lab07.calculatorwebservice.network.ClassifyNewsAsyncTask;

public class CalculatorWebServiceActivity extends AppCompatActivity {

    private EditText urlEditText;
    private TextView resultTextView;
    private TextView resultClass;
    private TextView resultConfidence;
    private Button displayResultButton;

    private DisplayResultButtonClickListener displayResultButtonClickListener = new DisplayResultButtonClickListener();
    private class DisplayResultButtonClickListener implements View.OnClickListener {

        @Override
        public void onClick(View view) {
            String newsUrl = urlEditText.getText().toString();
            ClassifyNewsAsyncTask calculatorWebServiceAsyncTask = new ClassifyNewsAsyncTask(resultTextView, resultClass, resultConfidence);
            calculatorWebServiceAsyncTask.execute(newsUrl);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_calculator_web_service);

        urlEditText = (EditText)findViewById(R.id.operator1_edit_text);
        resultTextView = (TextView)findViewById(R.id.result_text_view);
        resultClass = (TextView)findViewById(R.id.result_class);
        resultConfidence = (TextView)findViewById(R.id.result_confidence);

        displayResultButton = (Button)findViewById(R.id.display_result_button);
        displayResultButton.setOnClickListener(displayResultButtonClickListener);
    }
}

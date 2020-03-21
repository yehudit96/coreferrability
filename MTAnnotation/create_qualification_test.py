import boto3


def main():
    """
    create qualification test for the annotation task
    """
    SANDBOX = False
    if SANDBOX:
        sandbox_ep = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
        mturk = boto3.client('mturk', region_name='us-east-1', endpoint_url=sandbox_ep)

    else:
        mturk = boto3.client('mturk', region_name='us-east-1')

    with open('questions.xml', 'r') as q_f:
        question = q_f.read()

    with open('answer_key.xml') as f_in:
        answer_key = f_in.read()

    qual_response = mturk.create_qualification_type(
        Name='Paraphrases alignment',
        Description='This worker is qualified to answer HITs about English language semantics.',
        QualificationTypeStatus='Active',
        Test=question,
        AnswerKey=answer_key,
        TestDurationInSeconds=600
    )
    print(qual_response['QualificationType']['QualificationTypeId'])


if __name__ == '__main__':
    main()

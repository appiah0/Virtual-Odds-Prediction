import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabase = createClient(supabaseUrl, supabaseServiceKey);

const ADMIN_SECRET = process.env.ADMIN_SECRET_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { image, secret } = req.body;

  if (!secret || secret !== ADMIN_SECRET) {
    return res.status(401).json({ error: 'Unauthorized: Invalid passkey' });
  }

  if (!image) {
    return res.status(400).json({ error: 'Image data missing' });
  }

  try {
    const aiResponse = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-haiku-5-5',
        max_tokens: 2000,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'image',
                source: {
                  type: 'base64',
                  media_type: 'image/jpeg',
                  data: image.split(',')[1] || image
                }
              },
              {
                type: 'text',
                text: `Analyze this SportyBet virtual football results screenshot. Extract ALL matches belonging ONLY to the "Spain" and "Germany" leagues. Ignore all other leagues. For each Spain and Germany match, extract the home team name, away team name, pre-match 1X2 odds, and final scores. Reply with valid JSON only in this exact format, with no markdown code blocks:
                {"matches": [{"league": "Spain" or "Germany", "home": string, "away": string, "odds": {"home": number, "draw": number, "away": number}, "home_goals": number, "away_goals": number}]}`
              }
            ]
          }
        ]
      })
    });

    const aiData = await aiResponse.json();
    if (!aiResponse.ok) {
      throw new Error(aiData.error?.message || 'AI vision extraction failed');
    }

    const rawText = aiData.content[0].text.trim();
    const parsedData = JSON.parse(rawText);
    const matches = parsedData.matches || [];

    if (matches.length === 0) {
      return res.status(422).json({ error: 'No Spain or Germany matches found in image.' });
    }

    for (const m of matches) {
      const hGoals = Number(m.home_goals);
      const aGoals = Number(m.away_goals);
      const outcome = hGoals > aGoals ? 'home' : (hGoals === aGoals ? 'draw' : 'away');

      await supabase.from('virtual_round_results').insert({
        league: m.league,
        home: m.home,
        away: m.away,
        home_odds: Number(m.odds.home),
        draw_odds: Number(m.odds.draw),
        away_odds: Number(m.odds.away),
        home_goals: hGoals,
        away_goals: aGoals,
        outcome: outcome
      });
    }

    return res.status(200).json({ success: true, imported: matches.length });

  } catch (err) {
    console.error('Ingestion error:', err);
    return res.status(500).json({ error: err.message || 'Internal server error' });
  }
}
